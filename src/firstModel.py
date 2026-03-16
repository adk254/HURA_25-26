# ------------------
# Model Design v4
# ------------------
# Offspring strata: S, I (no movement)
# Adult strata:     R_a (cleared), R_c (chronic carrier) - movement between sites
#
# Maturation fates for I offspring (three branches):
#   1. Disease death      -> DEATH (exogenous sink)
#   2. Chronic carrier    -> R_c   (compartment, can move)
#   3. Cleared            -> R_a   (compartment, can move)
#
# Potential addition to make the biological assumptions more accurate:
#   Maturation fate for S offspring:
#   -> R_a (cleared adult, can move)
#       - This could be added into the 
#
# Births: proportional to total adult population (R_a + R_c), with vertical transmission
# Deaths: natural mortality out of every compartment

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from epymorph.kit import *  # noqa
from pathlib import Path
from sympy import Max

# TODO rework maturation - end of season
# fork I into dead/not dead, then EOS move all remaining I to R_c or R_a based on their infection status
# make movement, beta, maturation, birth, winter death functions of time 
#       use class system as seen in vignette 4 - time-variant beta

# breeding occurs at the beginning of the season
# migration occurs before the season

# ----------------------
# Load custom pond data
# ----------------------
current_dir = Path("firstModel.py").resolve().parent
test_data_path = current_dir / "data" / "basicTestData.csv"
df = pd.read_csv(test_data_path)

site_ids = df["site_id"].astype(str).tolist()

scope = CustomScope(site_ids)

# total_pop and adult_pop are static ints created initially.
total_pop = np.array(df["n_salamanders"].tolist(), dtype=int)

seed_location_index = 2
seed_size = int(df.loc[seed_location_index, "initial_infected"])

# one year timeframe
time = TimeFrame.of("2020-01-01", duration_days=365)

adult_frac = 0.40
adult_pop = np.floor(total_pop * adult_frac).astype(int)

# -------------------------
# make changes to shift from seeding into offspring to seeding into adults - keep this for later when we do multi-year simulations and want to seed into offspring
offspring_pop = (total_pop - adult_pop).astype(int)

# -------------------------
# Variables for Seasonality
# -------------------------

# season start and end variables
season_start = 135 # May 15
season_end = 258 # September 15

# --------------------------------
# Offspring IPM: S, I + deaths
# --------------------------------
class OffspringSI(CompartmentModel):
    compartments = [
        compartment("S", "susceptible offspring"),
        compartment("I", "infected offspring"),
    ]

    requirements = [
        AttributeDef("beta", type=float, shape=Shapes.TxN,
                     comment="offspring transmission rate"),
        AttributeDef("death_rate", type=float, shape=Shapes.TxN,
                     comment="offspring natural mortality rate"),
        AttributeDef("disease_death_rate", type=float, shape=Shapes.TxN,
                     comment="disease-induced death rate during season")
    ]

    def edges(self, symbols: ModelSymbols) -> list[TransitionDef]:
        S, I = symbols.all_compartments
        beta, mu, disease_death_rate = symbols.all_requirements

        N = Max(1, S + I)

        return [
            # density-dependent transmission within offspring pool
            edge(S, I, rate=beta * S * I / N),

            # natural deaths from each compartment
            edge(S, DEATH, rate=mu * S),
            edge(I, DEATH, rate=mu * I),
            edge(I, DEATH, rate=disease_death_rate * I)
        ]

# ----------------------------------------
# Adult IPM: R_a (cleared), R_c (carrier)
# both can move; neither dies from disease
# ----------------------------------------
class AdultRaRc(CompartmentModel):
    compartments = [
        compartment("R_a", "cleared adult salamanders"),
        compartment("R_c", "chronic carrier adult salamanders"),
    ]

    requirements = [
        AttributeDef("death_rate", type=float, shape=Shapes.TxN,
                     comment="adult natural mortality rate"),
    ]

    def edges(self, symbols: ModelSymbols) -> list[TransitionDef]:
        R_a, R_c = symbols.all_compartments
        mu, = symbols.all_requirements

        return [
            edge(R_a, DEATH, rate=mu * R_a),
            edge(R_c, DEATH, rate=mu * R_c),
        ]

# ---------------------------
# Multi-strata model builder
# ---------------------------
class SIR_v4(MultiStrataRUMEBuilder):
    def __init__(self):
        self.strata = [
            GPM(
                name="offspring",
                ipm=OffspringSI(),
                mm=mm.No(),  # offspring cannot move between sites
                init=init.NoInfection(),
                
            ),
            GPM(
                name="adult",
                ipm=AdultRaRc(),
                mm=mm.No(),  # adults move between sites

                init=init.SingleLocation(
                    # initial compartment is what you dont want to seed
                    initial_compartment="R_a",
                    # infection compartment is where you DO want to seed
                    infection_compartment="R_c",
                    location=seed_location_index,
                    seed_size=seed_size,
                ),
            ),
        ]

        self.meta_requirements = [
            AttributeDef("mature_rate", type=float, shape=Shapes.TxN,
                         comment="end of season maturation"),
            AttributeDef("birth_rate", type=float, shape=Shapes.TxN,
                         comment="births per adult per day"),
            AttributeDef("p_vert", type=float, shape=Shapes.TxN,
                         comment="probability offspring are born infected (vertical transmission)"),
            AttributeDef("p_chronic", type=float, shape=Shapes.TxN,
                         comment="probability infected offspring become chronic carriers on maturation"),
            AttributeDef("p_disease_death", type=float, shape=Shapes.TxN,
                         comment="probability infected offspring die from disease on maturation"),
        ]

    def meta_edges(self, symbols: MultiStrataModelSymbols) -> list[TransitionDef]:
        S, I = symbols.strata_compartments("offspring")
        R_a, R_c = symbols.strata_compartments("adult")

        mature_rate, birth_rate, p_vert, p_chronic, p_disease_death = symbols.all_meta_requirements

        # total adult population drives births
        #N_adult = Max(1, R_a + R_c)

        # p_clear is the remaining fraction after disease death and chronic carrier
        # these three probabilities must sum to 1:
        #   p_disease_death + p_chronic + p_clear = 1
        #p_clear = 1 - p_chronic - p_disease_death

        return [
            # --- End of Season: All S and I become their respective 
            # end of season become R_a
            edge(S, R_a, rate=mature_rate * S),

            # I -> R into fork structure
            # TODO: fix the mature_rate because right now it isnt working the way I intended - just need all to move on one day rather than 1.0 or 0.0 being apart of the rates *might not be true
            fork(
                edge(I, DEATH, rate = p_disease_death * mature_rate * I),
                edge(I, R_c, rate = p_chronic * mature_rate *I),
                edge(I, R_a, rate = (1 - p_chronic) * mature_rate * I),
            ),

            # Susceptible births: all R_a births are susceptible, and (1-p_vert) of R_c births are susceptible
            edge(BIRTH, S, rate=birth_rate * (R_a + ((1 - p_vert) * R_c))),

            # Infected births: only p_vert fraction of R_c births
            edge(BIRTH, I, rate=birth_rate * (p_vert * R_c)),
        ]

# ---------------------------
# Seasonal transmission beta
# ---------------------------
class SeasonalBeta(ParamFunctionTimeAndNode):
    def evaluate1(self, day: int, node_index: int) -> float:
        beta_active = 0.12
        beta_off = 0.0

        # modular for multi-year simulations
        t_mod = day % 365

        if season_start <= t_mod <= season_end:
            return beta_active
        else:
            return beta_off
        
# ---------------
# Seasonal births
# ---------------
class SeasonalBirths(ParamFunctionTimeAndNode):
    def evaluate1(self, day: int, node_index: int) -> float:
        birth_season = 1/30
        birth_off = 0.0

        t_mod = day % 365

        if season_start <= t_mod <= season_start + 7:
            return birth_season
        else:
            return birth_off
        
# ---------------
# Seasonal deaths - assuming all juveniles become adults at the end of the season
# ---------------
class SeasonalDeaths(ParamFunctionTimeAndNode):
    def evaluate1(self, day:int, node_index: int) -> float:
        winter = 1 / 365        # only affects adults
        summer = 1 / (365 * 3)  # affects adults and juveniles

        t_mod = day % 365

        if t_mod >= season_end or t_mod < season_start:
            return winter
        else:
            return summer

"""    
# ------------------
# Seasonal migration - Issue: the shape of commuter_proportion is Scalar not TxN
# ------------------
class SeasonalMigration(ParamFunctionTimeAndNode):
    def evaluate1(self, day: int, nodE_index: int) -> float:
        movement_on = 0.20
        movement_off = 0.0

        t_mod = day % 365

        if season_start - 30 <= t_mod < season_start:
            return movement_on
        else:
            return movement_off
"""

# -------------------
# Seasonal maturation
# -------------------
class SeasonalMaturation(ParamFunctionTimeAndNode):
    def evaluate1(self, day: int, node_index: int) -> float:
        move_all = 1.0
        season = 0.0

        t_mod = day % 365

        if t_mod >= season_end - 1 and t_mod <= season_end + 1:
            return move_all
        else:
            return season

# ---------------
# Build the RUME
# ---------------
rume = SIR_v4().build(
    scope=scope,
    time_frame=time,
    params={
        # offspring IPM params
            # class function incorporates seasonality
        "gpm:offspring::ipm::beta": SeasonalBeta(),
        "gpm:offspring::ipm::death_rate": SeasonalDeaths(),
        "gpm:offspring::ipm::disease_death_rate": 1 / 365,

        # adult IPM params
        "gpm:adult::ipm::death_rate": SeasonalDeaths(),

        # meta params
        "meta::ipm::mature_rate": SeasonalMaturation(),
        "meta::ipm::birth_rate": SeasonalBirths(),
        "meta::ipm::p_vert": 0.40,
        "meta::ipm::p_chronic": 0.5,
        "meta::ipm::p_disease_death": 0.20,

        # populations per strata
        "gpm:offspring::init::population": 0,#offspring_pop.tolist(), # potentially do ParamFunctionTxN here so first year is at 0 then other years are dynamic
        "gpm:adult::mm::population": adult_pop.tolist(),
        "gpm:adult::init::population": adult_pop.tolist(),

        # adult movement
        "gpm:adult::mm::commuter_proportion": 0.20,
    },
)

"""
# View the transmission rates over time
beta_values = (
    SeasonalMaturation()
    .with_context(
        scope=rume.scope,
        time_frame=rume.time_frame,
    )
    .evaluate()
)

### GRAPH ###
fig, ax = plt.subplots()
ax.plot(beta_values)
ax.set(title="beta function", ylabel="beta", xlabel="days")
fig.tight_layout()
plt.show()
"""

# ----------------------
# Model diagram
# ----------------------
#fig = rume.ipm.diagram()

# -----------------
# Run a simulation
# -----------------
sim = BasicSimulator(rume)
with sim_messaging(live=False):
    out = sim.run(rng_factory=default_rng(5))

df_out = out.dataframe

ponds = out.rume.scope.node_ids

plt.figure(figsize=(12, 6))

for pond in ponds:

    pond_df = df_out[df_out["node"] == pond]

    ticks = pond_df["tick"].to_numpy()

    S_series = pond_df["S_offspring"].to_numpy()
    I_series = pond_df["I_offspring"].to_numpy()
    Ra_series = pond_df["R_a_adult"].to_numpy()
    Rc_series = pond_df["R_c_adult"].to_numpy()

    plt.plot(ticks, S_series, label=f"{pond} - S")
    plt.plot(ticks, I_series, label=f"{pond} - I")
    plt.plot(ticks, Ra_series, linestyle="--", label=f"{pond} - R_a")
    plt.plot(ticks, Rc_series, linestyle=":", label=f"{pond} - R_c")

plt.xlabel("Day")
plt.ylabel("Population")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# View total offspring and adults across all ponds to check maturation works correctly
    # should see all offspring become adults at the end of the season (day 258)
tot_offspring = df_out.groupby("tick")[["S_offspring","I_offspring"]].sum()
print(tot_offspring.loc[254:258])
tot_adults = df_out.groupby("tick")[["R_a_adult","R_c_adult"]].sum()
print(tot_adults.loc[254:259])

print(df_out.groupby("tick")[["S_offspring","I_offspring"]].sum().loc[250:260])

"""
# view columns for diagnostic purposes
for column in df_out.columns:
    print(f"\n{column}\n")
"""