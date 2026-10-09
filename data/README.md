# Datasets & Provenance

In accordance with repo standards, raw dataset binaries are not committed to Git.

### 1. NASA C-MAPSS Turbofan Engine Degradation
* **Source:** NASA Prognostics Center of Excellence (PCoE)
* **Downloaded:** 2026-10-08 from `<URL>` (extracted to `data/raw/cmapss/`; the zip was not kept)
* **Used in:** Phase 1 (EDA & Baselines) & Phase 2 (Predictive Maintenance / RUL)
* **Citation:** A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," PHM 2008. This is the reference given in the dataset's own `readme.txt`.
* **License:** the dataset readme states no license.
* **Integrity:** SHA-256 of each raw file is in `data/cmapss.sha256`. Check with `sha256sum -c data/cmapss.sha256`.
* **Format:** 26 columns, space-delimited, no header: `unit_number`, `time_cycles`, 3 operating settings, 21 sensors (names are defined in `RAW_COLUMNS` in `01-predictive-maintenance/src/data/cmapss_loader.py`). NASA's readme lists sensors up to 26, but the files contain 21 sensors.
* **Derived, not in the files:** `rul`, the remaining useful life target. It is computed here (piecewise linear, cap 125). The cap is our modeling choice, not part of the dataset.
* **Decisions about capping the test truth:** see `docs/evaluation_protocol.md` (not yet written).

#### Engine counts (NASA readme)
| Subset | Train engines | Test engines | Operating conditions | Fault modes |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 | 1 (HPC) |
| FD002 | 260 | 259 | 6 | 1 (HPC) |
| FD003 | 100 | 100 | 1 | 2 (HPC, Fan) |
| FD004 | 248 | 249 | 6 | 2 (HPC, Fan) |

FD001 and FD002 counts were re-checked with the loader. FD003 and FD004 counts come from NASA's readme and were not re-checked.

#### Verified on load (FD001, checked 2026-10-09)
| Check | Result |
|---|---|
| Train / test engines | 100 / 100 |
| Train rows / test rows | 20,631 / 13,096 |
| Train cycle counters contiguous 1..N per engine | yes |
| Train sorted by (unit, cycle) | yes |
| Nulls (train / test) | 0 / 0 |
| `RUL_FD001.txt` length equals test engine count | yes (100) |
| Train lifetime min / median / max (cycles) | 128 / 199 / 362 |
| Test length min / median / max (cycles) | 31 / 133 / 303 |
| True test RUL min / median / max | 7 / 86 / 145 |
| Test engines with true RUL > 125 | 11 of 100 |
| Train rows at the 125 cap | 39.4% |

FD002 was loaded: test engine count matches the RUL file, and test lengths are 21 / 132 / 367 (min / median / max). Contiguity and null checks were run on FD001 only.

### 2. Case Western Reserve University (CWRU) Bearing Data Center (planned, not downloaded or verified)
* **Source:** Case Western Reserve University Bearing Data Center
* **Direct Access:** `https://engineering.case.edu/bearingdatacenter`
* **Planned use:** Phase 2 (vibration signal processing, spectral analysis, fault classification)
* **Description (unverified):** 12k/48k Drive End and Fan End bearing vibration recordings under variable motor loads (0-3 hp).

### 3. SECOM Semiconductor Manufacturing Process (planned, not downloaded or verified)
* **Source:** UCI Machine Learning Repository
* **Direct Access:** `https://archive.ics.uci.edu/dataset/179/secom`
* **Planned use:** Phase 1 & Phase 2 (class-imbalanced tabular quality classification)
* **Description (unverified):** 1,567 manufacturing observations across 591 sensor signals with heavily skewed pass/fail outcomes.