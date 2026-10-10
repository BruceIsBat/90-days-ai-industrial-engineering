# Evaluation Protocol: NASA C-MAPSS Turbofan Degradation (FD001)
**Status: Draft v1**

---

## 1. Schema & Data Ingestion
- **Schema**: 26 operational telemetry columns defined by the NASA Ames PHM benchmark:
  - `unit_number` (int)
  - `time_cycles` (int)
  - 3 operating settings (`op_setting_1` to `op_setting_3`)
  - 21 sensor measurements (`sensor_1` to `sensor_21`)
- **Informative Subsets**: In FD001, sensors with zero variance (1, 5, 6, 10, 16, 18, 19) are stripped during feature modeling. The 14 informative sensors are: `2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21`.
- **Integrity Validation**:
  - Gaps in operational cycles (must start at 1 and be contiguous $1..N$ with zero missing steps).
  - Ragged trailing delimiter ghost columns are stripped strictly to 26 columns.
  - Zero null or non-finite values permitted in telemetry.
  - Test unit numbers must strictly form a contiguous sequence $1..N$ matching the rows of `RUL_FD00x.txt`.

---

## 2. Target Engineering & Test-Truth Rule
- **Piecewise Linear Target**: Degradation onset is assumed to begin after an initial healthy period. Run-to-failure training trajectories are labeled with piecewise linear Remaining Useful Life (RUL) capped at $R_{\max} = 125.0$ cycles:
  $$y_t = \min(125.0, T_i - t)$$
- **Test-Truth Rule**:
  - Primary metric evaluation compares predictions $\hat{y}$ against test ground truth $y^* = \min(125.0, R^*)$, where $R^*$ is the true residual life vector provided in `RUL_FD001.txt`.
  - Predictions are evaluated at the terminal observed cycle $t = T_j$ of each censored test engine.
  - **Sensitivity Reporting**: Uncapped RMSE against raw $R^*$ must be reported alongside capped RMSE as a secondary sensitivity check, but it does not determine the benchmark win/loss verdict.

---

## 3. Split Design (Zero-Leakage Trajectory Boundary)
- **Grouping**: Partitioning is strictly performed at the trajectory level (`unit_number`), never via row-level shuffling.
- **FD001 Ratios**:
  - Training units: 80 engines (80%).
  - Validation units: 20 engines (20%).
  - Test units: 100 benchmark engines (official NASA test set).
- **Enforcement**: Disjoint intersection ($\text{Train Units} \cap \text{Val Units} = \emptyset$). Windows of validation engines must never appear in training folds.

---

## 4. Normalization Isolation
- Feature scaling parameters (Min-Max or Standard Scaler statistics) must be fitted **strictly on the 80 training engines**.
- Scaling parameters are transformed out-of-sample onto validation and test trajectories.
- Fitting scalers on the full dataset or re-fitting per window is strictly prohibited.

---

## 5. Sequence Windowing & Padding Rules
- **Window Length ($L$)**: Fixed sequence length lookback (e.g., $L = 30$).
- **Boundary Isolation**: Sliding windows must never cross engine trajectory boundaries.
- **Short Trajectory Padding**: For test engines where observed lifetime $T_j < L$, sequences must be zero-padded or left-replicated up to $L$ cycles with a corresponding boolean attention mask.
- **Evaluation Cutoff**: For test engines, only the terminal window ending at $t = T_j$ is evaluated.

---

## 6. Evaluation Metrics

### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2}$$

### NASA Asymmetric Scoring Function (PHM08 Benchmark)
Let error $d_i = \hat{y}_i - y_i$ (Predicted RUL minus True RUL):
$$s_i = \begin{cases} \exp\left(-\frac{d_i}{13}\right) - 1 & \text{if } d_i < 0 \text{ (Early prediction: safe replacement, operational cost)} \\ \exp\left(\frac{d_i}{10}\right) - 1 & \text{if } d_i \ge 0 \text{ (Late prediction: catastrophic failure, severe penalty)} \end{cases}$$
$$\text{NASA Score} = \sum_{i=1}^{N} s_i$$

---

## 7. Statistical Evaluation & Bootstrap Rules
- **Seeds**: Multi-library determinism fixed at seeds `42, 43, 44, 45, 46`.
- **Paired Bootstrap Confidence Intervals**:
  - $B = 10{,}000$ bootstrap iterations sampled with replacement across the 100 test engines.
  - Paired error delta: $\Delta_b = \text{RMSE}_{\text{candidate}}^{(b)} - \text{RMSE}_{\text{baseline}}^{(b)}$.
  - Compute 95% two-sided percentile confidence interval $[\Delta_{2.5\%}, \Delta_{97.5\%}]$.

---

## 8. Minimum Effect & The Three Outcomes
- **Minimum Effect Threshold**: A candidate model must achieve a relative improvement of at least **5.0%** over the LightGBM baseline RMSE ($19.4891 \times 0.95 = 18.5146$ cycles or lower).
- **The Three Outcomes**:
  1. **Win**: Candidate achieves $\ge 5\%$ reduction in test RMSE AND the upper bound of the 95% bootstrap CI on $\Delta$ is strictly negative ($\Delta_{97.5\%} < 0$).
  2. **Inconclusive**: Candidate point estimate improves by $\ge 5\%$, but the 95% bootstrap CI crosses zero ($\Delta_{97.5\%} \ge 0$), or improvement is between $0\%$ and $5\%$.
  3. **Loss**: Candidate fails to outperform the baseline point estimate ($\text{RMSE}_{\text{candidate}} \ge \text{RMSE}_{\text{baseline}}$).

---

## 9. Deviations & Historical Ledger Reconciliation

Per our immutable ledger policy, historical rows in `results.csv` are never modified or purged. The following rows predate the finalized evaluation protocol:

- **Row 1 (`2026-10-09T14:06:34.125187+00:00`)**: Evaluated against uncapped ground truth with a single seed (`42`). Ran on an uncommitted working tree flagged with `37413d2-dirty`.
- **Row 2 (`2026-10-10T...`)**: Evaluated against uncapped ground truth with a single seed (`42`). The commit hash (`131254c`) was clean, but model files remained untracked (`??`), bypassing the git-dirty guard. Furthermore, the `strict` flag described PyTorch determinism rather than LightGBM runtime settings.

**Verdict**: Neither Row 1 nor Row 2 counts toward the pre-registered win/loss decision. They serve exclusively as empirical receipts of single-seed execution and determinism.