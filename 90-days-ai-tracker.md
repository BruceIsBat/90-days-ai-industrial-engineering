# 90 Days of AI for Industrial & Production Engineering

> **NO EVIDENCE, NO CLAIM.**
> Every claim in this repo comes with a receipt: a plot, a results table, a commit, or a failure shown openly.

**Author:** Ibrahim Keji Awotedu ([LinkedIn](https://linkedin.com/in/ibrahim-keji) | [GitHub](https://github.com/BruceIsBat))
**Window:** Oct 3 - Dec 31, 2026 (Day 0 = Oct 2, announcement)
**Follow along:** #90DaysOfAI #NoEvidenceNoClaim

---

## The big question

**Where does AI actually beat simple baselines in industrial and production engineering, and by how much?**

## The rules

1. Every question names a **baseline** and a **metric** before any experiment runs.
2. A win is defined **before** I see results.
3. Results are averaged over **3-5 seeds**. A single run proves nothing.
4. If a model does not beat the baseline, I publish that too.
5. New ideas go in the backlog, not into the current phase.

## The three projects

### Project 1: Predictive Maintenance (Oct 18 - Nov 11)
- **Question:** Do deep learning models (LSTM, CNN, Transformer) predict machine failure better than a tuned gradient-boosting model?
- **Data:** NASA C-MAPSS turbofan engines
- **Baseline:** Tuned gradient boosting
- **Metric:** RMSE and the NASA asymmetric score, averaged over 5 seeds
- **Win:** Deep model beats the baseline by more than seed-to-seed variation
- **Loss:** Difference is within that variation, or the baseline wins

### Project 2: Physics-Informed ML (Nov 12 - Dec 6)
- **Question:** When does adding physics to a model help: small data, noisy data, or predictions outside the training range?
- **Data:** A manufacturing process I model (tool wear, heat, or degradation)
- **Baseline:** The same model without physics
- **Metric:** Error at 10%, 25%, 50%, 100% of the data, plus error on out-of-range inputs
- **Win:** Physics model is clearly better in at least one condition
- **Loss:** It is no better anywhere (still reported)

### Project 3: Scheduling (Dec 7 - Dec 23)
- **Question:** Can a reinforcement learning scheduler beat standard dispatch rules and a solver, and does failure prediction improve the schedule?
- **Data:** Standard job-shop benchmark instances plus my own simulator
- **Baseline:** Dispatch rules (SPT, EDD, FIFO) and OR-Tools
- **Metric:** Makespan and tardiness
- **Win:** RL beats the rules on unseen instances, or failure prediction reduces delays
- **Loss:** RL only wins on instances like its training set

## Plan

| Phase | Dates | Theme |
|---|---|---|
| 1. Foundations and Data | Oct 3 - Oct 17 | Show Me the Data |
| 2. Predictive Maintenance and Quality | Oct 18 - Nov 11 | Prove It Works |
| 3. Physics-Informed ML | Nov 12 - Dec 6 | Physics Doesn't Lie |
| 4. Scheduling and Decisions | Dec 7 - Dec 23 | Beat the Rule |
| 5. Consolidate and Publish | Dec 24 - Dec 31 | Case Closed |

Day-by-day plan: see [`90-days-ai-tracker.md`](90-days-ai-tracker.md).

## Results so far

| Project | Baseline | Best model | Metric | Seeds | Verdict |
|---|---|---|---|---|---|
| 1. Predictive Maintenance | - | - | - | - | In progress |
| 2. Physics-Informed ML | - | - | - | - | Not started |
| 3. Scheduling | - | - | - | - | Not started |

Raw run logs: [`results.csv`](results.csv) (seed, config, metric for every run).

## Repo structure

```
.
├── README.md
├── 90-days-ai-tracker.md
├── results.csv
├── backlog.md
├── data/                  # download scripts and notes (no raw data committed)
├── 01-predictive-maintenance/
├── 02-physics-informed-ml/
├── 03-scheduling/
└── reports/               # weekly writeups and final report
```

## Datasets

- NASA C-MAPSS turbofan engine degradation simulation
- CWRU Bearing Data Center
- SECOM semiconductor manufacturing data (UCI)

Each dataset's source, license, and citation will be listed in `data/README.md` before it is used.

## Challenge my results

Think a baseline is weak? Think a gain is just noise? **Open an issue.** Public pushback is welcome and will be answered in the open.

## Reproducibility

Every experiment will include fixed seeds, pinned requirements, and a run script. If you cannot reproduce a result, that is a bug. Please report it.
