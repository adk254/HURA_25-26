# HURA_25-26

## Modelling Climate-Driven Ambystoma tigrinum Virus Dynamics  
### Integrating Temperature, Landscape Fragmentation, and Host Behaviour using Epymorph

This repository contains research code, simulations, figures, and supplemental materials for my Hooper Undergraduate Research Award (HURA) project at Northern Arizona University.

The project investigates how **climate change**, **temperature variation**, **habitat structure**, and **host movement behaviour** may influence the spread of **Ambystoma tigrinum virus (ATV)** in tiger salamander populations using a spatially explicit epidemiological modelling framework.

---

## Research Overview

Ambystoma tigrinum virus (ATV) is a ranavirus that infects tiger salamanders and can cause severe mortality events. Disease dynamics are influenced by environmental conditions such as:

- Water temperature  
- Seasonal timing  
- Host breeding movement  
- Connectivity among ponds  
- Habitat fragmentation  

This project uses **Epymorph**, a spatial disease-modelling package in Python, to simulate ATV dynamics across pond networks under varying environmental conditions.

---

## Current Model Features

### Within-Pond Disease Dynamics
Compartmental structure includes:

- `S_offspring` – susceptible larvae  
- `I_offspring` – infected larvae  
- `R_a_adult` – recovered adults  
- `R_c_adult` – chronic carrier adults  

### Biological Processes

- Horizontal transmission  
- Vertical transmission  
- Temperature-dependent transmission rates  
- Temperature-dependent mortality  
- Seasonal reproduction  
- End-of-season maturation  
- Adult movement between ponds

### Spatial Structure

Current simulations include:

- Three-pond landscapes  
- Connected pond networks  
- Movement during breeding season  
- Experimental fragmentation scenarios

---

## Ongoing / Future Work

- Regional temperature scenarios using PRISM climate data  
- Distinct temperature regimes across ponds  
- Sensitivity analyses  
- Expanded landscape networks  
- Climate warming scenarios  
- Additional validation against empirical ranavirus literature

---

## Repository Contents

```text
HURA_25-26/
│── models/              # Python simulation scripts
│── data/                # Input climate or test data
│── figures/             # Output figures and plots
│── README.md
