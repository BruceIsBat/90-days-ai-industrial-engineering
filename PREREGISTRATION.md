# Benchmark Pre-Registration: 90 Days of AI in Industrial Engineering

## Project 1: Remaining Useful Life (RUL) Estimation on NASA C-MAPSS FD001
- **Status**: Frozen
- **Date Registered**: Draft written 2026-10-09; frozen at the commit tagged prereg-v1
- **Benchmark Baseline**: LightGBM + Optuna (50 trials) on rolling temporal features
- **Baseline Performance**: **19.4891 RMSE** | **NASA Score: 1215.46** (Seed 42, FD001 capped at 125 cycles)

### Pre-Registered Hypotheses
We test whether deep sequence architectures (LSTM, 1D-CNN) outperform a tuned gradient-boosted tabular baseline when evaluated under strict trajectory isolation.

### Minimum Effect Size
- **Threshold**: Minimum **5.0% relative improvement** in primary test RMSE over the LightGBM baseline:
  $$\text{Target RMSE} \le 19.4891 \times (1 - 0.05) = 18.5146 \text{ cycles}$$

### Decision Criteria (The Three Outcomes)
Evaluated across 10,000 paired bootstrap resamples on the 100 terminal test engine predictions:
1. **Win**: Candidate model test RMSE $\le 18.5146$ AND the upper bound of the 95% bootstrap confidence interval of the error difference strictly excludes zero ($\Delta_{97.5\%} < 0$).
2. **Inconclusive**: Candidate improves test RMSE by $\ge 5\%$, but the bootstrap CI crosses zero ($\Delta_{97.5\%} \ge 0$), or point estimate improvement is between $0\%$ and $5\%$.
3. **Loss**: Candidate model test RMSE is worse than or equal to the LightGBM baseline ($19.4891$).

### Compute & Hyperparameter Parity
- Neural architectures receive an identical tuning budget of 50 Optuna trials using the Tree-structured Parzen Estimator (TPE) sampler.
- Training stopped early if validation loss fails to improve for 20 consecutive epochs.

---

## Project 2: Physics-Informed Neural Networks for Chemical Reactor Thermal Dynamics
- **Status**: Draft, revise before start date
- **Target System**: Continuous Stirred-Tank Reactor (CSTR) non-linear dynamics
- **Baseline**: Standard numerical ODE solver (Runge-Kutta 4th order) & unconstrained MLP
- **Hypothesis**: PINN loss formulation preserves thermodynamic energy conservation bounds under noisy sensor regimes.

---

## Project 3: Reinforcement Learning for Job Shop Scheduling (JSSP)
- **Status**: Draft, revise before start date
- **Target Benchmark**: Taillard JSSP benchmark instances
- **Baseline**: Priority Dispatching Rules (FIFO, SPT, MWKR) & OR-Tools CP-SAT solver
- **Hypothesis**: PPO agent achieves lower makespan ($C_{\max}$) within 100ms inference time budgets compared to heuristic dispatching.