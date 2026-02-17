# ------------------
# Model Design v3
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

# ----------------------
# Load custom pond data
# ----------------------
current_dir = Path("firstModel.py").resolve().parent
test_data_path = current_dir / "data" / "basicTestData.csv"
df = pd.read_csv(test_data_path)

site_ids = df["site_id"].astype(str).tolist()
scope = CustomScope(site_ids)

total_pop = np.array(df["n_salamanders"].tolist(), dtype=int)

seed_location_index = 2
seed_size = int(df.loc[seed_location_index, "initial_infected"])

time = TimeFrame.of("2020-01-01", duration_days=200)

adult_frac = 0.40
adult_pop = np.floor(total_pop * adult_frac).astype(int)
offspring_pop = (total_pop - adult_pop).astype(int)

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
    ]

    def edges(self, symbols: ModelSymbols) -> list[TransitionDef]:
        S, I = symbols.all_compartments
        beta, mu = symbols.all_requirements

        N = Max(1, S + I)

        return [
            # density-dependent transmission within offspring pool
            edge(S, I, rate=beta * S * I / N),

            # natural deaths from each compartment
            edge(S, DEATH, rate=mu * S),
            edge(I, DEATH, rate=mu * I),
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
class SIR_v3(MultiStrataRUMEBuilder):
    def __init__(self):
        self.strata = [
            GPM(
                name="offspring",
                ipm=OffspringSI(),
                mm=mm.No(),  # offspring cannot move between sites
                init=init.SingleLocation(
                    location=seed_location_index,
                    seed_size=seed_size
                ),
            ),
            GPM(
                name="adult",
                ipm=AdultRaRc(),
                mm=mm.Flat(),  # adults move between sites
                init=init.NoInfection(initial_compartment="R_a"),
            ),
        ]

        self.meta_requirements = [
            AttributeDef("mature_rate", type=float, shape=Shapes.TxN,
                         comment="rate at which offspring mature into adults"),
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
        N_adult = Max(1, R_a + R_c)

        # p_clear is the remaining fraction after disease death and chronic carrier
        # these three probabilities must sum to 1:
        #   p_disease_death + p_chronic + p_clear = 1
        p_clear = 1 - p_chronic - p_disease_death

        return [
            # Potential addition of biological assumption that Susceptible offspring 
            # can metamorphose without becoming infected
            # edge(S, R_a, rate=mature_rate * S),

            # --- I offspring maturation (three fates) ---
            # fate 1: disease death on maturation
            edge(I, DEATH, rate=p_disease_death * mature_rate * I),

            # fate 2: become chronic carrier adult
            edge(I, R_c, rate=p_chronic * mature_rate * I),

            # fate 3: clear infection on maturation, become cleared adult
            edge(I, R_a, rate=p_clear * mature_rate * I),

            # --- births driven by total adult population ---
            edge(BIRTH, S, rate=(1 - p_vert) * birth_rate * N_adult),
            edge(BIRTH, I, rate=p_vert * birth_rate * N_adult),
        ]

# ---------------
# Build the RUME
# ---------------
rume = SIR_v3().build(
    scope=scope,
    time_frame=time,
    params={
        # offspring IPM params
        "gpm:offspring::ipm::beta": 0.30,                    # placeholder
        "gpm:offspring::ipm::death_rate": 1 / (365 * 3),    # placeholder: 3yr lifespan

        # adult IPM params
        "gpm:adult::ipm::death_rate": 1 / (365 * 3),

        # meta params
        "meta::ipm::mature_rate": 1 / 60,
        "meta::ipm::birth_rate": 1 / 120,
        "meta::ipm::p_vert": 0.60,
        "meta::ipm::p_chronic": 0.30,
        "meta::ipm::p_disease_death": 0.20,
        # note: p_clear = 1 - p_chronic - p_disease_death = 0.50 (implicit)

        # populations per strata
        "gpm:offspring::mm::population": offspring_pop.tolist(),
        "gpm:offspring::init::population": offspring_pop.tolist(),
        "gpm:adult::mm::population": adult_pop.tolist(),
        "gpm:adult::init::population": adult_pop.tolist(),

        # adult movement
        "gpm:adult::mm::commuter_proportion": 0.20,
    },
)

# ----------------------
# Model diagram
# ----------------------
fig = rume.ipm.diagram()

# -----------------
# Run a simulation
# -----------------
sim = BasicSimulator(rume)
with sim_messaging(live=False):
    out = sim.run(rng_factory=default_rng(5))

df_out = out.dataframe

# view columns for diagnostic purposes
for column in df_out.columns:
    print(f"\n{column}\n")