# ------------------
# Model Design
# ------------------
# Offspring strata: S, I (no movement)
# Adult Strata:     R (movement between sites)
# Births: proportional to adult R, with vertical transmission
# Deaths: out of every compartment
# Maturation: I_offspring -> R_adult

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from epymorph.kit import * #noqa

from pathlib import Path
from sympy import Max

# load custom pond data
current_dir = Path("firstModel.py").resolve().parent
print(current_dir, current_dir.parent)
test_data_path = current_dir / "data" / "basicTestData.csv"
df = pd.read_csv(test_data_path)

# create custom scope using site_ids
site_ids = df["site_id"].astype(str).tolist()
scope = CustomScope(site_ids)

total_pop = np.array(df["n_salamanders"].tolist(), dtype=int)

# infection seed (offspring I)
seed_location_index = 2
seed_size = int(df.loc[seed_location_index, "initial_infected"])

time = TimeFrame.of("2020-01-01", duration_days=200)

# split total population into two strata (the choice made for this model)
adult_frac = 0.40 # adjust as needed
adult_pop = np.floor(total_pop * adult_frac).astype(int)
offspring_pop = (total_pop - adult_pop).astype(int)

#-----------------------------
# Offspring IPM: S, I + deaths
#-----------------------------
class OffspringSI(CompartmentModel):
    compartments = [
        compartment("S", "susceptible offspring"),
        compartment("I", "infected offspring"),
    ]

    requirements = [
        AttributeDef("beta", type=float, shape=Shapes.TxN, 
                     comment="offspring transmission rate"),
        AttributeDef("death_rate", type=float, shape=Shapes.TxN,
                     comment="offspring mortality rate"),
    ]

    def edges(self, symbols: ModelSymbols) -> list[TransitionDef]:
        S, I = symbols.all_compartments
        beta, mu = symbols.all_requirements

        N = Max(1, S + I)  # avoid division by zero

        return [
            # density dependent transmission inside offspring pool
            edge(S, I, rate=beta * S * I / N),

            # deaths from each compartment
            edge(S, DEATH, rate=mu * S),
            edge(I, DEATH, rate=mu * I),
        ]

#-----------------------------
# Adult IPM: R ONLY and deaths
#-----------------------------
class AdultR(CompartmentModel):
    compartments = [
        compartment("R", "adult salamanders"),
    ]

    requirements = [
        AttributeDef("death_rate", type=float, shape=Shapes.TxN,
                     comment="adult mortality rate"),
    ]

    def edges(self, symbols: ModelSymbols) -> list[TransitionDef]:
        R, = symbols.all_compartments
        mu, = symbols.all_requirements

        return [
            edge(R, DEATH, rate=mu * R),
        ]

#---------------------------
# Multi-strata model builder
#---------------------------
class SIR_v1(MultiStrataRUMEBuilder):
    def __init__(self):
        # define two strata
        self.strata = [
            GPM(
                name = "offspring",
                ipm=OffspringSI(),
                mm=mm.No(), # offspring cannot move between sites
                init = init.SingleLocation(location=seed_location_index, seed_size = seed_size),
            ),
            GPM(
                name="adult",
                ipm=AdultR(),
                mm=mm.Flat(), # just a basic flat movement model for now
                init=init.NoInfection(initial_compartment = "R"),
            ),
        ]

        # parameters
        self.meta_requirements = [
            AttributeDef("mature_rate", type=float, shape=Shapes.TxN),
            AttributeDef("birth_rate", type=float, shape=Shapes.TxN),
            AttributeDef("p_vert", type=float, shape=Shapes.TxN)
        ]

    def meta_edges(self, symbols: MultiStrataModelSymbols) -> list[TransitionDef]:
        # get componenets by their strata
        S, I = symbols.strata_compartments("offspring")
        R, = symbols.strata_compartments("adult")

        # meta parameters
        mature_rate, birth_rate, p_vert = symbols.all_meta_requirements

        return [
            # maturation - infected offspring become adults
            edge(I, R, rate=mature_rate * I),

            # births split into both S and I offspring via vertical transmission
            edge(BIRTH, S, rate = (1 - p_vert) * birth_rate * R),
            edge(BIRTH, I, rate = p_vert * birth_rate * R),
        ]
    
#---------------
# Build the RUME
#---------------
rume = SIR_v1().build(
    scope = scope,
    time_frame = time,
    params = {
        # offspring IPM params
        "gpm:offspring::ipm::beta": 0.30, # placeholeder
        "gpm:offspring::ipm::death_rate": 1 / (365 * 3), # placeholder - 3 year avg. lifespan

        # adult IPM params
        "gpm:adult::ipm::death_rate": 1 / (365 * 3), # placeholder

        # meta params
        "meta::ipm::mature_rate": 1 / 60, # placeholder - avg. 60 days to maturation
        "meta::ipm::birth_rate": 1 / 120, # placeholder - births per adult per day
        "meta::ipm::p_vert": 0.60, # placeholder - 60% infected through birth

        # populations per strata
        "gpm:offspring::mm::population": offspring_pop.tolist(),
        "gpm:offspring::init::population": offspring_pop.tolist(),
        "gpm:adult::mm::population": adult_pop.tolist(),
        "gpm:adult::init::population": adult_pop.tolist(),

        # adult flat movement
        "gpm:adult::mm::commuter_proportion": 0.20, # 20% chance adults move
    },
)

#-----------------
# Run a simulation
#-----------------
sim = BasicSimulator(rume)
with sim_messaging(live=False):
    out = sim.run(rng_factory=default_rng(5))

df_out = out.dataframe

# filter to not show duplicates
df_out = df_out.groupby(df_out.columns, axis = 1).sum()
# view columns for diagnostic purposes
for column in df_out.columns:
    print(f"\n{column}\n")