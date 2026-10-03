# 90 Days of AI for Industrial & Production Engineering
**Theme:** NO EVIDENCE, NO CLAIM | **Hashtags:** #90DaysOfAI #NoEvidenceNoClaim
**Rule:** every post carries a receipt (plot, table, commit, or a shown failure). 3-4 hrs/day. Saturday = skeptic review with Claude. Light days double as recovery buffers.

**Daily post template:**
```
Day X/90 | [Phase name]

Claim: [one line]
Receipt: [plot / table / commit link]
Caveat: [what didn't work or is uncertain]

#90DaysOfAI #NoEvidenceNoClaim
```

## Day 0 - Fri Oct 2, 2026: ANNOUNCEMENT
- [ ] Publish the manifesto post (LinkedIn + X). Pin it. Repo link in first comment.


## Phase 1 - Foundations and Data: "Show Me the Data" (Oct 3 - Oct 17)

| Done | Day | Date | Task | Post |
|---|---|---|---|---|
| [x] | 1 | Sat Oct 03 | Create public repo + folder structure; README with the rule 'No Evidence, No Claim'; set up environment. Done: seeds.py + 9 passing tests. Moved to Day 2: run_logger.py + metrics.py | LAUNCH follow-up: pin the manifesto, share the repo link. |
| [ ] | 2 | Sun Oct 04 | Survey AI in IPE (predictive maintenance, quality, scheduling, digital twins); write a one-page map. Carry-over: run_logger.py + metrics.py with tests | Sunday: weekly longer writeup (LinkedIn article / thread). |
| [ ] | 3 | Mon Oct 05 | Pick 2 core datasets; write the evaluation protocol (splits, metrics, baseline rules); draft PREREGISTRATION.md from the revised project questions | Monday: THE CLAIM. State what you're testing this week. |
| [ ] | 4 | Tue Oct 06 | Time-series refresher: windowing, rolling features | Daily receipt (plot / table / commit). |
| [ ] | 5 | Wed Oct 07 | Validation for time-dependent data; demo data leakage on a toy example | Wednesday: RECEIPTS. Mid-week chart. |
| [ ] | 6 | Thu Oct 08 | Tabular refresher: gradient boosting, tuning discipline | Daily receipt (plot / table / commit). |
| [ ] | 7 | Fri Oct 09 | Catch-up day + Week 1 writeup | Friday: EXPLAIN EXPLAIN teardown. Take a type of hype claim, show the evidence needed. |
| [ ] | 8 | Sat Oct 10 | EDA: NASA C-MAPSS turbofan data | Saturday: daily receipt + send week's results to Claude for skeptic review. |
| [ ] | 9 | Sun Oct 11 | EDA: CWRU bearing vibration data | Sunday: weekly longer writeup (LinkedIn article / thread). |
| [ ] | 10 | Mon Oct 12 | EDA: SECOM (missing values, imbalance) | Monday: THE CLAIM. State what you're testing this week. |
| [ ] | 11 | Tue Oct 13 | EDA: Tennessee Eastman process (skim) or alternative dataset | Daily receipt (plot / table / commit). |
| [ ] | 12 | Wed Oct 14 | Baseline 1: linear + gradient boosting on C-MAPSS | Wednesday: RECEIPTS. Mid-week chart. |
| [ ] | 13 | Thu Oct 15 | Baseline 2: gradient boosting on SECOM | Daily receipt (plot / table / commit). |
| [ ] | 14 | Fri Oct 16 | Baseline 3: simple LSTM on C-MAPSS | Friday: EXPLAIN EXPLAIN teardown. Take a type of hype claim, show the evidence needed. |
| [ ] | 15 | Sat Oct 17 | Baseline results table + Phase 1 recap; commit and tag PREREGISTRATION.md (frozen before Oct 18) | Saturday: daily receipt + send week's results to Claude for skeptic review. |

## Phase 2 - Predictive Maintenance and Quality: "Prove It Works" (Oct 18 - Nov 11)

| Done | Day | Date | Task | Post |
|---|---|---|---|---|
| [ ] | 16 | Sun Oct 18 | RUL: preprocessing, capped RUL targets (cap 125), per-regime normalization for FD002 | Sunday: weekly longer writeup (LinkedIn article / thread). |
| [ ] | 17 | Mon Oct 19 | RUL: sliding windows + evaluation (RMSE, asymmetric score); splits grouped by engine | Monday: THE CLAIM. State what you're testing this week. |
| [ ] | 18 | Tue Oct 20 | RUL: tuned gradient boosting benchmark (rolling features, 50 Optuna trials) | Daily receipt (plot / table / commit). |
| [ ] | 19 | Wed Oct 21 | RUL: LSTM model | Wednesday: RECEIPTS. Mid-week chart. |
| [ ] | 20 | Thu Oct 22 | RUL: LSTM tuning (same 50-trial budget), multiple seeds | Daily receipt (plot / table / commit). |
| [ ] | 21 | Fri Oct 23 | RUL: 1D-CNN model | Friday: EXPLAIN EXPLAIN teardown. Take a type of hype claim, show the evidence needed. |
| [ ] | 22 | Sat Oct 24 | RUL: Transformer encoder model | Saturday: daily receipt + send week's results to Claude for skeptic review. |
| [ ] | 23 | Sun Oct 25 | RUL: model comparison across 5 seeds; primary deep model chosen on validation, one test run, paired bootstrap CI over engines | Sunday: weekly longer writeup (LinkedIn article / thread). |
| [ ] | 24 | Mon Oct 26 | RUL: ablations (window size, feature sets) | Monday: THE CLAIM. State what you're testing this week. |
| [ ] | 25 | Tue Oct 27 | RUL: error analysis by engine and degradation stage | Daily receipt (plot / table / commit). |
| [ ] | 26 | Wed Oct 28 | Fault diagnosis: load and segment CWRU signals | Wednesday: RECEIPTS. Mid-week chart. |
| [ ] | 27 | Thu Oct 29 | Spectral features: FFT, envelope analysis | Daily receipt (plot / table / commit). |
| [ ] | 28 | Fri Oct 30 | Wavelet features | Friday: EXPLAIN EXPLAIN teardown. Take a type of hype claim, show the evidence needed. |
| [ ] | 29 | Sat Oct 31 | Classical ML classifier on engineered features | Saturday: daily receipt + send week's results to Claude for skeptic review. |
| [ ] | 30 | Sun Nov 01 | 1D-CNN on raw vibration + Day 30 milestone scorecard | Day 30 milestone thread: scorecard (shipped, baselines beaten, baselines NOT beaten). |
| [ ] | 31 | Mon Nov 02 | Anomaly detection: autoencoder | Monday: THE CLAIM. State what you're testing this week. |
| [ ] | 32 | Tue Nov 03 | Anomaly detection: threshold selection, false-alarm analysis | Daily receipt (plot / table / commit). |
| [ ] | 33 | Wed Nov 04 | Uncertainty: deep ensembles | Wednesday: RECEIPTS. Mid-week chart. |
| [ ] | 34 | Thu Nov 05 | Uncertainty: conformal prediction intervals | Daily receipt (plot / table / commit). |
| [ ] | 35 | Fri Nov 06 | Calibration check: coverage plots | Friday: EXPLAIN EXPLAIN teardown. Take a type of hype claim, show the evidence needed. |
| [ ] | 36 | Sat Nov 07 | Explainability: SHAP on RUL / quality model | Saturday: daily receipt + send week's results to Claude for skeptic review. |
| [ ] | 37 | Sun Nov 08 | SECOM quality prediction: class imbalance handling | Sunday: weekly longer writeup (LinkedIn article / thread). |
| [ ] | 38 | Mon Nov 09 | SECOM: modeling + feature selection (light day) | Monday: THE CLAIM. State what you're testing this week. |
| [ ] | 39 | Tue Nov 10 | Cost-based evaluation: maintenance cost vs failure cost (light day) | Daily receipt (plot / table / commit). |
| [ ] | 40 | Wed Nov 11 | Project 1 writeup + repo cleanup (light day) | Wednesday: RECEIPTS. Mid-week chart. |

## Phase 3 - Physics-Informed ML: "Physics Doesn't Lie" (Nov 12 - Dec 6)

| Done | Day | Date | Task | Post |
|---|---|---|---|---|
| [ ] | 41 | Thu Nov 12 | PINN basics: set up a simple ODE / heat-equation PINN | Daily receipt (plot / table / commit). |
| [ ] | 42 | Fri Nov 13 | PINN: loss weighting experiments | Friday: EXPLAIN EXPLAIN teardown. Take a type of hype claim, show the evidence needed. |
| [ ] | 43 | Sat Nov 14 | PINN: reproduce and document failure modes | Saturday: daily receipt + send week's results to Claude for skeptic review. |
| [ ] | 44 | Sun Nov 15 | PINN: reproduce a published simple example | Sunday: weekly longer writeup (LinkedIn article / thread). |
| [ ] | 45 | Mon Nov 16 | Pick ONE dataset (real tool-wear data, or synthetic with deliberate truth-vs-prior mismatch); define data plan and baselines (data-driven NN + empirical wear curve) | Monday: THE CLAIM. State what you're testing this week. |
| [ ] | 46 | Tue Nov 17 | Build the physics model of the process | Daily receipt (plot / table / commit). |
| [ ] | 47 | Wed Nov 18 | Generate/clean data with realistic noise | Wednesday: RECEIPTS. Mid-week chart. |
| [ ] | 48 | Thu Nov 19 | Pure data-driven baseline on small data | Daily receipt (plot / table / commit). |
| [ ] | 49 | Fri Nov 20 | PINN v1 on the process | Friday: EXPLAIN EXPLAIN teardown. Take a type of hype claim, show the evidence needed. |
| [ ] | 50 | Sat Nov 21 | PINN debugging | Saturday: daily receipt + send week's results to Claude for skeptic review. |
| [ ] | 51 | Sun Nov 22 | PINN tuning (same budget as the baseline) | Sunday: weekly longer writeup (LinkedIn article / thread). |
| [ ] | 52 | Mon Nov 23 | Small-data sweep: 10%, 25%, 50%, 100% of data | Monday: THE CLAIM. State what you're testing this week. |
| [ ] | 53 | Tue Nov 24 | Noise robustness test | Daily receipt (plot / table / commit). |
| [ ] | 54 | Wed Nov 25 | Extrapolation test beyond training range (train first 70% of life, test last 30%) | Wednesday: RECEIPTS. Mid-week chart. |
| [ ] | 55 | Thu Nov 26 | PINN vs both baselines: paired CI of the error difference | Daily receipt (plot / table / commit). |
| [ ] | 56 | Fri Nov 27 | Hybrid 1: physics-derived features in standard models | Friday: EXPLAIN EXPLAIN teardown. Take a type of hype claim, show the evidence needed. |
| [ ] | 57 | Sat Nov 28 | Hybrid 2: residual learning on top of the physical model | Saturday: daily receipt + send week's results to Claude for skeptic review. |
| [ ] | 58 | Sun Nov 29 | Hybrid comparison | Sunday: weekly longer writeup (LinkedIn article / thread). |
| [ ] | 59 | Mon Nov 30 | Ablation: which physics terms matter | Monday: THE CLAIM. State what you're testing this week. |
| [ ] | 60 | Tue Dec 01 | Day 60 milestone scorecard + analysis: when does physics help? | Day 60 milestone thread: scorecard. |
| [ ] | 61 | Wed Dec 02 | Write the 'when does physics help?' analysis | Wednesday: RECEIPTS. Mid-week chart. |
| [ ] | 62 | Thu Dec 03 | Figures and tables for Project 2 | Daily receipt (plot / table / commit). |
| [ ] | 63 | Fri Dec 04 | Project 2 writeup draft | Friday: EXPLAIN EXPLAIN teardown. Take a type of hype claim, show the evidence needed. |
| [ ] | 64 | Sat Dec 05 | Fix issues from skeptic review | Saturday: daily receipt + send week's results to Claude for skeptic review. |
| [ ] | 65 | Sun Dec 06 | Project 2 release | Sunday: weekly longer writeup (LinkedIn article / thread). |

## Phase 4 - Scheduling and Decisions: "Beat the Rule" (Dec 7 - Dec 23)

| Done | Day | Date | Task | Post |
|---|---|---|---|---|
| [ ] | 66 | Mon Dec 07 | Scheduling basics: job-shop formulation | Monday: THE CLAIM. State what you're testing this week. |
| [ ] | 67 | Tue Dec 08 | Implement dispatch rules (SPT, MWKR, FIFO) | Daily receipt (plot / table / commit). |
| [ ] | 68 | Wed Dec 09 | Load benchmark instances (Taillard / OR-Library) | Wednesday: RECEIPTS. Mid-week chart. |
| [ ] | 69 | Thu Dec 10 | OR-Tools CP-SAT baseline (same time budget as RL inference) | Daily receipt (plot / table / commit). |
| [ ] | 70 | Fri Dec 11 | Metrics: makespan, tardiness; benchmark table | Friday: EXPLAIN EXPLAIN teardown. Take a type of hype claim, show the evidence needed. |
| [ ] | 71 | Sat Dec 12 | SimPy shop-floor simulator | Saturday: daily receipt + send week's results to Claude for skeptic review. |
| [ ] | 72 | Sun Dec 13 | Gym environment wrapper | Sunday: weekly longer writeup (LinkedIn article / thread). |
| [ ] | 73 | Mon Dec 14 | RL design: state, action, reward. Define numeric convergence criterion + RL checkpoint (fallback decision) | Monday: THE CLAIM. State what you're testing this week. |
| [ ] | 74 | Tue Dec 15 | RL: PPO training v1 | Daily receipt (plot / table / commit). |
| [ ] | 75 | Wed Dec 16 | RL debugging | Wednesday: RECEIPTS. Mid-week chart. |
| [ ] | 76 | Thu Dec 17 | RL tuning (if not converged: switch to priority-index rules, reported as a fallback) | Daily receipt (plot / table / commit). |
| [ ] | 77 | Fri Dec 18 | Compare RL vs dispatch rules vs OR-Tools (Question 3a outcome table) | Friday: EXPLAIN EXPLAIN teardown. Take a type of hype claim, show the evidence needed. |
| [ ] | 78 | Sat Dec 19 | Generalization to unseen instance sizes | Saturday: daily receipt + send week's results to Claude for skeptic review. |
| [ ] | 79 | Sun Dec 20 | Disruptions: machine breakdowns | Sunday: weekly longer writeup (LinkedIn article / thread). |
| [ ] | 80 | Mon Dec 21 | Question 3b: health-aware rescheduling (hold the dispatcher fixed, vary only RUL information) | Monday: THE CLAIM. State what you're testing this week. |
| [ ] | 81 | Tue Dec 22 | Test the integrated pipeline | Daily receipt (plot / table / commit). |
| [ ] | 82 | Wed Dec 23 | Phase 4 writeup (light day) | Wednesday: RECEIPTS. Mid-week chart. |

## Phase 5 - Consolidate and Publish: "Case Closed" (Dec 24 - Dec 31)

| Done | Day | Date | Task | Post |
|---|---|---|---|---|
| [ ] | 83 | Thu Dec 24 | Repo cleanup, READMEs | Daily receipt (plot / table / commit). |
| [ ] | 84 | Fri Dec 25 | Reproducibility: seeds, requirements, run scripts | Friday: EXPLAIN EXPLAIN teardown. Take a type of hype claim, show the evidence needed. |
| [ ] | 85 | Sat Dec 26 | Unify results tables across projects | Saturday: daily receipt + send week's results to Claude for skeptic review. |
| [ ] | 86 | Sun Dec 27 | Tests + CI | Sunday: weekly longer writeup (LinkedIn article / thread). |
| [ ] | 87 | Mon Dec 28 | Technical report draft | Monday: THE CLAIM. State what you're testing this week. |
| [ ] | 88 | Tue Dec 29 | Report revision with Claude review | Daily receipt (plot / table / commit). |
| [ ] | 89 | Wed Dec 30 | Portfolio page + final figures | Wednesday: RECEIPTS. Mid-week chart. |
| [ ] | 90 | Thu Dec 31 | Final retrospective + scorecard + year-end post | Day 90 milestone thread: final scorecard + retrospective. |

## Backlog (new ideas go here, not into the current phase)
-