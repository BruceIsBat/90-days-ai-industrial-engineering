# Pre-Registration of Hypotheses & Experimental Boundaries
**Project Theme:** NO EVIDENCE, NO CLAIM  
**Date Registered:** October 8, 2026

## Anchor Research Question
*Where does AI actually beat simple, rigorous baselines in industrial and production engineering, and by how much?*

---

## Project 1: Machine Degradation & RUL Estimation (Oct 18 – Nov 11)
* **Research Question:** Do deep sequence models (LSTM, 1D-CNN, Transformer) achieve lower Remaining Useful Life (RUL) regression error than a tuned Gradient Boosting baseline on turbofan engine degradation?
* **Benchmark Scope:** NASA C-MAPSS dataset (FD001 primary, FD002 secondary).
* **Baseline Standard:** Tuned LightGBM regressor trained on rolling temporal feature windows (sizes 5, 10, 20).
* **Evaluation Threshold:** Evaluated across 5 random seeds (42, 43, 44, 45, 46) using 95% paired bootstrap confidence intervals over test engines.
* **Win Criteria:** The deep model achieves lower RMSE than tuned LightGBM, and the 95% bootstrap confidence interval of the error difference does not cross zero.
* **Loss Criteria:** The 95% bootstrap confidence interval includes zero, or LightGBM achieves an equal or lower RMSE.

---

## Project 2: Physics-Informed ML & Inductive Biases (Nov 12 – Dec 6)
* **Research Question:** Under what operational boundary conditions does regularizing a model with physical laws improve performance: small training data (10%–50%), high sensor noise ($\sigma = 20\%$), or out-of-distribution operating range extrapolation?
* **Benchmark Scope:** Semi-empirical degradation/tool wear model with simulated stochastic disturbance.
* **Baseline Standards:** 
  1. Purely data-driven neural network (identical architecture without the physical loss penalty).
  2. Regularized empirical parametric curve fit (Taylor tool-life model).
* **Win Criteria:** The physics-informed formulation yields at least a 20% relative reduction in prediction error ($L_2$) during out-of-distribution extrapolation tests with non-overlapping 95% intervals.
* **Loss Criteria:** The physical prior yields $<20\%$ error reduction during extrapolation, or the classical parametric formula outperforms neural approaches in small-data regimes.

---

## Project 3: Production Operations & Dynamic Scheduling (Dec 7 – Dec 23)
* **Question 3a (Static Efficiency):** Can an RL policy (PPO) beat deterministic priority dispatch rules (SPT, EDD, FIFO) on unseen job-shop instances, coming within 5% makespan of OR-Tools CP-SAT under a 1-second solve budget?
  * **Win:** RL policy generalizes to unseen instance sizes with a lower makespan than the best dispatch rule.
  * **Loss:** RL policy fails to beat Shortest Processing Time (SPT) on unseen benchmark instances.
* **Question 3b (Health-Aware Rescheduling):** Does integrating Remaining Useful Life degradation predictions into dispatch queues reduce total tardiness compared to reactive breakdown repairs and fixed maintenance intervals?
  * **Win:** Proactive scheduling with noisy degradation forecasts reduces machine tardiness by $>15\%$ compared to both reactive dispatch and periodic maintenance under breakdown shocks.
  * **Loss:** The improvement in tardiness is $<15\%$, or periodic maintenance achieves equivalent throughput.

---

## Pre-Registered Commitments
1. Criteria are locked before running sweeps.
2. Results are logged to `results.csv` regardless of whether they confirm or refute the hypothesis. Negative results will be reported with full ablation details.
