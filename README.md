# Rings of Presence

**A simpler picture of gravity, tested against real data.**

This repository holds the code behind the research paper *Rings of Presence: one balance rule for gravitational boundaries, and the galactic acceleration scale as c²√(Λ/3)/2π* (Hassan Anwar, 2026; arXiv link to follow) and the article *Rings of Presence: A Simpler Picture of Gravity, Tested Against Real Data* (on Medium; link to follow).

Every object is a ring, and its presence fills the inside. Presence never stops at an edge, so every ring encloses every other. These scripts turn that picture into formulas and test each one against public measurements: GPS clocks, Mercury's orbit, a binary pulsar, ocean tides, the moons of six planets, 175 galaxy rotation curves, 111 satellite galaxies, the Local Group and 1,590 supernovae.

## What it is, and what it isn't

- **Where tested, it reproduces known physics exactly.** The full set of rules is equivalent to Einstein's general relativity plus standard cosmology. Its value is a simpler way to see the same physics, not better accuracy.
- **One rule replaces four.** `r³ = G·m / S` gives the Hill sphere, the tidal radius, the Roche lobe and the turnaround radius.
- **One new proposal:** a reason for the factor 2π in the long-noticed relation a₀ ≈ c²√(Λ/3)/2π (Milgrom 1999). The total presence inside a ring, divided by its edge value and size, is 2π for a flat ring and for a ball, so a₀ = 0.86–0.94 × 10⁻¹⁰ m/s² with nothing adjustable. It was found after looking at the data, so it still has to be tested on new data.
- **About a dozen ideas failed** and are reported as failures in the article.

## The formulas

```
Presence           P = Σ G·E / (c⁴·r)                  E = M·c² for matter
Three readings     value of P   -> clock rates
                   slope of P   -> pull g = G·M / r²
                   change of the pull across an object -> stretch (tides) = 2·G·M·d / D³
Ownership ring     r³ = G·m / S     S = 3·G·M / D³ inside a host,  S = Λ·c² / 3 inside emptiness
Galaxy pattern     g_obs = g_bar / (1 − e^(−√(g_bar / a₀)))
a₀ from the ring   a₀ = c²·√(Λ / 3) / k,   k = ∫ P dV / (P_edge · Rⁿ) = 2π for a flat ring (n = 2) and a ball (n = 3)
                   ->  a₀ = 0.86 (Planck) – 0.94 (SH0ES) × 10⁻¹⁰ m/s²
                   (earlier scripts k_ring_test.py and k_derivation_test.py use the older form k = π(1 + x),
                    which mixed a 2-D area with a 3-D average; it is kept for the record)
Superposition      τ = ħ / E_G,   E_G ≈ (6/5)·G·m² / R
```

The article has the full side-by-side comparison with standard formulas and a guide to every symbol.

## Quick start

```bash
pip install -r requirements.txt
bash get_data.sh     # downloads the Pantheon+ supernova data (about 33 MB) into ./data
bash run_all.sh      # runs every test and saves each script's output in ./results
```

The SPARC rotation curves and the Local Volume Database tables are already included in `data/` (both are openly licensed). Only the supernova data is downloaded: its release has no license file, so it isn't redistributed here. Without it, every test except `expansion_tests.py` still runs.

Tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Astropy 8.0. The full run takes about 3 minutes.

Each script also runs on its own. Pass it the data folder it needs (for SPARC, the folder holding the `*_rotmod.dat` files), for example:

```bash
python3 scripts/fit_a0.py data/SPARC/Rotmod_LTG
python3 scripts/expansion_tests.py "data/pantheon_plus/Pantheon+_Data/4_DISTANCES_AND_COVAR"
```

## The paper

`paper/` holds the research paper (PDF and LaTeX source with figures). Every number in it comes from `paper_analysis.py` and `paper_additions.py`.

## The scripts

| Script | What it tests | Data |
|---|---|---|
| `presence_model_tests.py` | light bending, radar delay, GPS, Mercury's orbit (simulated), fifth-dimension limit, light pulling light | physical constants |
| `presence_round2.py` | presence, pull and stretch at Earth from every source; clock tests; three-body numbers; superposition lifetimes | constants, published measurements |
| `presence_judge.py` | who owns an object: its owner's pull against the stretch from everything else | constants |
| `delayed_presence_binary.py` | a binary where presence arrives late, with four choices of where the pull aims | Hulse–Taylor pulsar |
| `fixed_binary.py` | the Hulse–Taylor pulsar over 1,000 orbits with every (v/c)², (v/c)⁴ and (v/c)⁵ term | published masses and orbit |
| `fit_a0.py` | scans a₀ against 2,803 measurements in 164 galaxies | SPARC |
| `galaxy_outside_pull.py` | each galaxy fitted with its own outside pull | SPARC; neighbour pulls from Chae et al. 2020 |
| `outside_presence_tests.py` | wide star pairs, galaxy outskirts, Crater II, the double slit, presence as an observer | SPARC; published measurements |
| `k_ring_test.py` | k = π(1 + x): what x the galaxies need, and whether a₀ changes with galaxy size | SPARC |
| `k_derivation_test.py` | x from the ring's shape (ball or flat); gas-rich galaxies; a variant using each galaxy's mass layout | SPARC |
| `satellite_predictions.py` | star speeds in 73 satellite galaxies, predicted before the measured speeds are loaded | Local Volume Database |
| `satellite_history.py` | 42 satellites traced back 3 billion years through the Milky Way and the Large Magellanic Cloud | Local Volume Database, Gaia motions |
| `sand_fit.py` | a "bumping" (self-interacting) dark-matter law fitted to satellites, with a shuffle test | Local Volume Database |
| `glass_tests.py` | heating from star formation against dark-matter centre density; fast movers | Read, Walker & Steger 2019; Local Volume Database |
| `ultrafaint_centres.py` | faint satellites' centres against untouched dark matter | Local Volume Database |
| `ring_tests.py` | planets' rings and their moons; galaxies' rings and satellites; the Local Group's ring from velocities | JPL; Local Volume Database |
| `expansion_tests.py` | 1,590 supernovae: emptiness against six alternatives, with the full error matrix | Pantheon+ |
| `paper_analysis.py` | every number and figure in the research paper: ring rule checks, the a₀ fits with bootstrap, intrinsic scatter and mass-to-light systematics, the gas-rich subsample and the size test | SPARC, Local Volume Database |
| `paper_additions.py` | ladder of rings, a₀ at redshift 0.65–2.45 (41 disks from Price et al. 2021), robustness and visible-shape checks | SPARC, published high-redshift values |

`fit_a0.py`, `delayed_presence_binary.py` and `satellite_predictions.py` also write CSV files (`a0_scan.csv`, `binary_separation.csv`, `satellite_predictions.csv`) into the folder they run from.

## Main results

| Test | Predicted | Measured |
|---|---|---|
| GPS satellite clocks | +38.51 µs per day | +38.58 µs per day built in |
| Mercury's extra orbit turning | 42.98″ per century | about 43″ |
| Light bending at the Sun's edge | 1.751″ | 1.75″ |
| Hulse–Taylor orbit decay | 1.0001 × Einstein | 0.9983 ± 0.0016 × Einstein |
| Tides: Sun's share against the Moon's | 0.46 | 0.46 |
| Atom clocks 1 mm apart | 1.09 × 10⁻¹⁹ | (0.98 ± 0.23) × 10⁻¹⁹ |
| Farthest moons of six planets | inside their rings | 0.02 to 0.46 of the way out |
| Local Group's ring | at most 1.4–1.9 Mpc | 0.96 ± 0.03 Mpc |
| Supernova distances | need emptiness | emptiness 66.8 ± 1.8%; alternatives worse by Δχ² = 48 to 1,902 |
| a₀ (× 10⁻¹⁰ m/s²) | 0.86 (Planck) – 0.94 (SH0ES) | 0.82 (+0.22/−0.17) from 16 gas-rich galaxies; 1.14 ± 0.08 from all 164 at the usual star weight, matching at a 20% heavier one |
| a₀ against redshift | constant | no rise from z = 0.65 to 2.45 in 41 disks (evolution not yet excluded) |
| a₀ against galaxy properties | no dependence | none with size, distance, brightness, speed, gas or bulge fraction |

The `results/` folder holds the full output of every script from the run reported in the article.

## Data and credits

- **SPARC** rotation curves (included in `data/SPARC`): Lelli, McGaugh & Schombert 2016, *AJ* 152, 157. CC BY 4.0; please cite the paper if you use the data.
- **Local Volume Database** tables (included in `data/local_volume_database`): A. B. Pace, [github.com/apace7/local_volume_database](https://github.com/apace7/local_volume_database), commit `bb444a7`. CC0 1.0.
- **Pantheon+** supernovae (downloaded by `get_data.sh` from [github.com/PantheonPlusSH0ES/DataRelease](https://github.com/PantheonPlusSH0ES/DataRelease)): Scolnic et al. 2022, *ApJ* 938, 113; Brout et al. 2022, *ApJ* 938, 110.
- **Dark matter at 150 pc**: Read, Walker & Steger 2019 (values typed into `glass_tests.py`).
- **Moon orbits**: NASA JPL Solar System Dynamics, planetary satellite mean elements.
- Other published measurements are cited in the scripts' comments and in the article.

## How this was made

The ideas and the questions behind each step are Hassan Anwar's. The formulas, code and analysis were developed with Claude, an AI model by Anthropic, as a research assistant, and checked against published measurements. This is not peer-reviewed work. Corrections and challenges are welcome: please open an issue.

## License

The code is under the MIT License (see `LICENSE`). Each dataset keeps its own license and citation requirements.
