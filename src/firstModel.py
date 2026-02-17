# ------------------
# Model Design
# ------------------
# Offspring strata: S, I (no movement)
# Adult strata:     R_a (cleared), R_c (chronic carrier) - movement between sites
# Births: proportional to adult population, with vertical transmission
# Deaths: out of every compartment
# Maturation: S_offspring -> R_a, I_offspring -> R_a or R_c

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
                     comment="offspring mortality rate"),
    ]

    def edges(self, symbols: ModelSymbols) -> list[TransitionDef]:
        S, I = symbols.all_compartments
        beta, mu = symbols.all_requirements

        N = Max(1, S + I)

        return [
            edge(S, I, rate=beta * S * I / N),
            edge(S, DEATH, rate=mu * S),
            edge(I, DEATH, rate=mu * I),
        ]

# ----------------------------------------
# Adult IPM: R_a (cleared), R_c (carrier)
# ----------------------------------------
class AdultRaRc(CompartmentModel):
    compartments = [
        compartment("R_a", "cleared adult salamanders"),
        compartment("R_c", "chronic carrier adult salamanders"),
    ]

    requirements = [
        AttributeDef("death_rate", type=float, shape=Shapes.TxN,
                     comment="adult mortality rate"),
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
class SIR_v2(MultiStrataRUMEBuilder):
    def __init__(self):
        self.strata = [
            GPM(
                name="offspring",
                ipm=OffspringSI(),
                mm=mm.No(),
                init=init.SingleLocation(
                    location=seed_location_index,
                    seed_size=seed_size
                ),
            ),
            GPM(
                name="adult",
                ipm=AdultRaRc(),
                mm=mm.Flat(),
                init=init.NoInfection(initial_compartment="R_a"),
            ),
        ]

        self.meta_requirements = [
            AttributeDef("mature_rate", type=float, shape=Shapes.TxN),
            AttributeDef("birth_rate", type=float, shape=Shapes.TxN),
            AttributeDef("p_vert", type=float, shape=Shapes.TxN),
            AttributeDef("p_chronic", type=float, shape=Shapes.TxN,
                         comment="probability infected offspring become chronic carriers on maturation"),
        ]

    def meta_edges(self, symbols: MultiStrataModelSymbols) -> list[TransitionDef]:
        S, I = symbols.strata_compartments("offspring")
        R_a, R_c = symbols.strata_compartments("adult")

        mature_rate, birth_rate, p_vert, p_chronic = symbols.all_meta_requirements

        # total adult population for births
        N_adult = Max(1, R_a + R_c)

        return [
            # I offspring either clear on maturation or become chronic carriers
            edge(I, R_a, rate=(1 - p_chronic) * mature_rate * I),
            edge(I, R_c, rate=p_chronic * mature_rate * I),

            # births driven by total adult population
            edge(BIRTH, S, rate=(1 - p_vert) * birth_rate * N_adult),
            edge(BIRTH, I, rate=p_vert * birth_rate * N_adult),
        ]

# ---------------
# Build the RUME
# ---------------
rume = SIR_v2().build(
    scope=scope,
    time_frame=time,
    params={
        "gpm:offspring::ipm::beta": 0.30,
        "gpm:offspring::ipm::death_rate": 1 / (365 * 3),
        "gpm:adult::ipm::death_rate": 1 / (365 * 3),

        "meta::ipm::mature_rate": 1 / 60,
        "meta::ipm::birth_rate": 1 / 120,
        "meta::ipm::p_vert": 0.60,
        "meta::ipm::p_chronic": 0.40,

        "gpm:offspring::mm::population": offspring_pop.tolist(),
        "gpm:offspring::init::population": offspring_pop.tolist(),
        "gpm:adult::mm::population": adult_pop.tolist(),
        "gpm:adult::init::population": adult_pop.tolist(),

        "gpm:adult::mm::commuter_proportion": 0.20,
    },
)

# ----------------------
# Diagram
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