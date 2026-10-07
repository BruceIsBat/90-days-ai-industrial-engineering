# Backlog

New ideas and known gaps go here, not into the current phase.
Rule: an item leaves this file only when it is done (commit linked) or deliberately dropped (reason written).

Format: `- [ ] item | why it matters | revisit by`

---

## Seeds (`common/seeds.py`)
- [ ] CUDA paths untested (CPU only so far) | no GPU reproducibility claim is allowed until tested | before the first GPU run (Phase 2, Colab/Kaggle)
- [ ] Add a "warn" determinism mode (strict / warn / off) | strict mode may block some GPU models, and the ledger already distinguishes strict / warn_only / none | when strict mode first raises on GPU
- [ ] Decide whether `np.int64` seeds are accepted (currently TypeError), then add a test documenting the choice | seeds from NumPy arrays or pandas columns | Day 4
- [ ] Functional test for `get_worker_init_fn` (only its guard is tested) | proves it seeds the right values | Day 4
- [ ] Docstring note: the worker init function must be used together with `generator=get_torch_generator(seed)` on the DataLoader | worker randomness is reproducible only with both | Day 4
- [ ] Run a formatter (ruff or black) on common/ and tests/ | consistent style | Day 86 (cleanup)

## Run logger (`common/run_logger.py`)
- [ ] Diverged runs leave no record in the ledger (NaN/inf are rejected) | project rule is to publish failures; avoid survivorship bias. Option: `status` column (ok / diverged) | before Project 1 training (Oct 18)
- [ ] Decide on `config` serialization: `default=str` silently stringifies NumPy values, Paths, and objects | configs must be comparable across runs; leaning toward failing loudly | before Project 1 (Oct 18)
- [ ] Determinism mode is read at log time, not run time | call `log_experiment` while the same settings are active, or capture the mode at run start | before Project 1
- [ ] `device` is a typed argument | derive it from the model (`next(model.parameters()).device.type`) in training code | before Project 1
- [ ] Commit hash is taken at log time, not run start | edits during a long run can mislabel it; capture at start and pass `commit_hash` | before long training runs
- [ ] Concurrent writes to `results.csv` can interleave | matters if runs are parallelized | when parallel runs start
- [ ] Wrong-type text fields (e.g. `project=5`) raise ValueError; consider TypeError | clearer error | Day 4
- [ ] `config` is not type-checked; `notes=None` writes an empty cell | input hygiene | Day 4
- [ ] Safeguard test: committed `results.csv` header equals `CSV_HEADERS` | the header has drifted twice already | Day 4

## Logger and conftest tests
- [ ] Redirect test: log with no `filepath`, row lands in the redirected file, real `results.csv` byte-identical; then mutation check (delete the monkeypatch line, test must fail) | proves the safety net works | Day 4
- [ ] Remaining bad-input cases: empty/whitespace strings, string or None metric, `seed=3.5`, `seed=True`, integer `project` | coverage | Day 4
- [ ] Environment-column tests: determinism mode (strict / warn_only / none), device, torch and numpy versions | backs the reproducibility contract | Day 4
- [ ] conftest: the determinism fixture restores only one setting (not `warn_only` or the cuDNN flags); remove leftover imports from the old test file | state leakage on GPU | before GPU work

## Git hash tests (`tests/test_git_hash.py`)
- [ ] Case: `results.csv` AND `code.py` both modified, result must be dirty | guards against an exclusion that hides real changes | Day 4
- [ ] Clean-state test should assert the exact hash, not just "not unknown" | stronger check | Day 4
- [ ] Staged and deleted changes | documents what "dirty" means | Day 4
- [ ] `skipif` when git is missing; `-c commit.gpgsign=false`; `GIT_CEILING_DIRECTORIES` for the non-repo case | portability and CI | before CI (Day 86)
- [ ] Mutation checks: remove the exclude entry, then `--untracked-files=no`; the matching test must fail each time | shows the tests can fail | Day 4
- [ ] Remove or trim the older mock-based hash test if it still exists | redundant | Day 4

## Other modules
- [ ] `common/metrics.py`: RMSE with a hand-computed test, mismatched lengths, empty arrays | needed before any model results | Day 4
- [ ] `common/splits.py`: review (not yet seen) | leakage risk | before Phase 2 (Oct 18)

## Repo
- [ ] Make `common` installable (pyproject.toml) | scripts in subfolders like `01-predictive-maintenance/` cannot import it today, and folder names starting with digits are not Python packages | before Project 1 (Oct 18)
- [ ] CI (run pytest on push) | the receipt for every claim | Day 86
- [ ] Mutation-check receipts: keep the red and green screenshots for the seeds and git hash checks | post material | Day 4

## Research plan (carry into `PREREGISTRATION.md`, freeze by Day 15)
- [ ] Project 1: choose the primary deep model on validation only; test set run once; require a minimum practical effect (e.g. 5% RMSE) as well as a CI excluding zero; same maximum lookback for all models; per-regime normalization for FD002
- [ ] Project 2: pick ONE dataset (real tool-wear data or synthetic with deliberate truth-vs-prior mismatch); baselines = data-driven NN + empirical wear curve (not Taylor tool-life); align question and win condition; define "out-of-range" and noise sigma
- [ ] Project 3a: replace EDD with MWKR (Taillard has no due dates); one time limit used consistently; outcome table covering all four RL-vs-rules / RL-vs-solver combinations
- [ ] Project 3b: hold the dispatcher fixed and vary only RUL information; define predictor quality as a sweep; add a disruption metric; RL checkpoint with a numeric convergence criterion on Day 73

## Ideas (unvetted)
- [ ] 
