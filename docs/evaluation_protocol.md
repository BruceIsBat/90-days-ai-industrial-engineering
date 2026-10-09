# Evaluation Protocol: Prognostics & Degradation

## 1. Schema & Column Assignment
The C-MAPSS raw space-delimited text files contain 26 unnamed columns:
- Col 0: `unit_number` (Engine ID)
- Col 1: `time_cycles` (Operational cycle index)
- Col 2-4: `op_setting_1`, `op_setting_2`, `op_setting_3`
- Col 5-25: `sensor_1` through `sensor_21`

## 2. Partitioning Strategy (Zero-Leakage Trajectory Splits)
- Training split: Subdivided into 80% train engines and 20% validation engines using `split_trajectories_by_unit(seed=42)`.
- Temporal Boundary: Partitions are grouped strictly by `unit_number`. No sequence window or temporal slice from any training engine may appear in the validation partition.
- Sensor Normalization: Scaling (Z-score or MinMax) parameters must be calculated strictly on the 80% train split and applied out-of-sample to validation and test partitions.

## 3. Label Definition & Piecewise Linear Transformation
- Piecewise RUL Ceiling: Remaining Useful Life is capped at `max_rul = 125.0` cycles.
- Run-to-Failure Training Engines: For engine $i$ with maximum observed life $T_i$, the RUL at cycle $t$ is:
  $$y_t = \min(125, T_i - t)$$
- Truncated Test Engines: For test engine $j$ truncated at cycle $T_j$ with true remaining life $R_j^*$ provided in `RUL_FD00x.txt`:
  $$y_t = \min(125, (T_j - t) + R_j^*)$$
  Evaluation takes place strictly on the terminal window ($t = T_j$) of each test engine.

## 4. Evaluation Metrics
- Primary Metric: Root Mean Squared Error (RMSE) on test engines:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}$$
- Secondary Metric: NASA Asymmetric Scoring Function (penalizes late predictions exponentially):
  $$S = \sum_{i=1}^N s_i, \quad s_i = \begin{cases} \exp\left(-\frac{\hat{y}_i - y_i}{13}\right) - 1 & \text{if } \hat{y}_i - y_i < 0 \text{ (Late prediction)} \\ \exp\left(\frac{\hat{y}_i - y_i}{10}\right) - 1 & \text{if } \hat{y}_i - y_i \ge 0 \text{ (Early prediction)} \end{cases}$$

## 5. Baseline Rules & Parity
- Feature Parity: Tabular models (LightGBM) receive rolling temporal aggregates (mean, standard deviation over windows [5, 10, 20]) rather than isolated raw cycles.
- Tuning Parity: Both baseline and neural models are allotted identical hyperparameter search budgets (50 trials with fixed seeds via Optuna).
