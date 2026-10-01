# RL Course + Project

The project contains three CartPole exercise notebooks and the trading workflow. The tuning section has executed results and saved artifacts. PDF course material is unchanged.

## Separate PPO project

Open **[Project_PPO.ipynb](Project_PPO.ipynb)** for the PPO implementation and expanded BTC/USDT experiment.

Training starts **April 1, 2020 UTC**; the cache includes seven preceding days for causal feature warm-up. Evaluation retains August 17–September 17, 2026 (end exclusive). `btc_data.py` downloads official Binance monthly spot archives, verifies SHA-256 checksums, handles the 2025 timestamp-unit change, checks overlap with the existing cache, and records provenance. The archives omit 24 hours; these gaps are recorded, never filled with synthetic candles. Features restart after gaps and training episodes use only continuous hourly windows. No API key is required.

All seven PPO candidates use nine discrete target exposures:
`[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]`.
Environment actions are integer indices 0–8; cash is index 4. Negative exposures short BTC and positive exposures hold BTC.

Run All validates the data and performs a 2,048-transition training/checkpoint smoke test. Full tuning is opt-in with `RETRAIN_ALL=True`, or the command below. New artifacts go to `eval/ppo_202004_nine_actions/`. The saved results in `eval/ppo/` are historical results for the previous training period/action sets and do not apply to this configuration. The DQN notebook and artifacts remain separate.

```sh
.venv-rl/bin/python btc_data.py
.venv-rl/bin/python -m unittest discover -s tests -p 'test_ppo.py'
.venv-rl/bin/python tests/test_btc_data.py
.venv-rl/bin/python ppo_trading.py all --workers 3
```

The expanded-history experiment has completed **67 training fits** (28 screening, 36 confirmation, 3 final fits; 2,375,680 training transitions). Validation selected **`ppo_baseline` with unanimous voting**. It uses the same fitted networks as the mean-probability baseline, with a different aggregation rule.

| Policy | Evaluation return | Maximum drawdown |
|---|---:|---:|
| Selected PPO, unanimous vote | -0.57% | 0.57% |
| Baseline PPO, mean probability | 14.44% | 4.02% |
| Buy and hold | 21.11% | 7.71% |
| Cash | 0.00% | 0.00% |

The selected policy held cash for 742 of 743 decisions. Tuning reduced drawdown but **did not improve evaluation returns**. Even its validation selection score was below the cash benchmark. Results: [report](eval/ppo_202004_nine_actions/results_summary.md), [evaluation CSV](eval/ppo_202004_nine_actions/evaluation_results.csv), [selected checkpoint](eval/ppo_202004_nine_actions/final_ppo_agent.pth). The notebook contains executed results and independently reproduces the saved policy's return.

`build_ppo_notebook.py` regenerates the notebook from the Python implementation. `ppo_report.py` reports the newly trained policy. Screening uses seven configurations, four chronological validation folds, and 16,384 transitions per fit. Confirmation uses three seeds and 49,152 transitions per fit. The assignment month is a reused holdout, not an independent unseen test. These measurements do not establish an advantage from the longer dataset; the original and new experiments also differ in action space.

## Current trading policy

Tuning improved the fixed assignment-period return from **−20.37% to −6.04%**, a **14.33 percentage-point improvement**. Maximum drawdown fell from **29.44% to 8.44%**. The tuned policy still lost money and underperformed cash and fee-adjusted buy-and-hold (**+21.11%**).

The policy uses momentum and volatility features, Double DQN, scaled training rewards, soft target updates, and a turnover penalty. Three independently trained networks must agree on a position; otherwise it holds cash. Selection used four chronological validation windows before the assignment evaluation period. That evaluation period was viewed previously, so this is a reused holdout comparison, not a fresh independent test.

- `Project_Instruction.ipynb`: original experiments followed by the executed tuning section and results.
- `trading_tuning.py`: training, validation, ensemble agreement rules, checkpoints, and evaluation.
- `tuning_report.py`: comparison with the original model, benchmarks, charts, and canonical submission files.
- `eval/results_summary.md`: full measured comparison and limitations.
- `presentation_notes.md`: updated presentation outline and speaking notes.

## Run

Select the **RL Python 3.12** Jupyter kernel, or create a Python 3.12 environment and install `requirements.txt`. The verified local interpreter is `.venv-rl/bin/python`.

The new notebook section loads the completed tuning artifacts by default. Set `RETRAIN_TUNING=True` to repeat the full search, or run:

```sh
.venv-rl/bin/python trading_tuning.py screen --workers 3
.venv-rl/bin/python trading_tuning.py confirm --workers 3
.venv-rl/bin/python trading_tuning.py aggregate
.venv-rl/bin/python trading_tuning.py final --workers 3
.venv-rl/bin/python tuning_report.py
```

Screening compares seven configurations across four folds. Confirmation trains the strongest two with three seeds and twice the screening budget. A documented follow-up compares four agreement rules using those validation checkpoints. Final fitting uses all data before 2026-08-17 with the chosen settings and three seeds. The search includes 52 validation training runs and three final fits. Completed trials and matching final fits resume from cache. Data and implementation fingerprints prevent silently mixing incompatible runs; use a new output directory for a changed protocol.

## Current outputs

- `eval/evaluation_results.csv`: current ten-row submission, fixed 2026-08-17 to 2026-09-17 UTC window.
- `eval/final_trading_agent.pth`: current ensemble, including its agreement rule, feature order, and normalization.
- `eval/selected_configuration.json`: current selected settings.
- `eval/tuning/`: complete search results, per-seed models, validation benchmarks, comparison plot, final trace, and reproducibility protocols.
- `eval/original_baseline/`: preserved original model, CSV, configuration, and results.

Reload the current policy with the loader that preserves its agreement rule:

```python
from trading_tuning import ValidatedEnsemblePolicy, load_data, add_features
agent = ValidatedEnsemblePolicy.load('eval/final_trading_agent.pth')
features = add_features(load_data(), agent.config.features)
# Use make_env(features_for_period, agent.positions) to supply observations.
```

Ten deterministic episodes on the same history are reproducibility checks, not independent performance samples. `eval/tuning/initial_mean_q/` preserves an intermediate mean-Q artifact that was computed but not reviewed during agreement-rule selection.

## Verification

```sh
.venv-rl/bin/python tests/test_agents.py
.venv-rl/bin/python tests/test_trading.py
.venv-rl/bin/python tests/test_tuning.py
```

Trading implementation tests use synthetic fixtures only. Reported returns always use real cached Binance data. Checks cover causal features, training-only normalization, replay, target updates, agreement rules, checkpoint reloads, and deterministic evaluation.

## CartPole notebooks

1. `TD1_Question.ipynb`: tabular Q-learning and discretization.
2. `TD2_Question.ipynb`: online neural Q-learning and the gradient correction.
3. `TD3_Question.ipynb`: replay DQN and Double DQN.

Each algorithm trained for 1,000 episodes with seed 23 and used the same five held-out reset seeds for greedy evaluation.

| Algorithm | Mean evaluation reward |
|---|---:|
| Tabular Q-learning | 132.4 |
| Basic online DQN | 325.8 |
| Replay DQN, width 128 | 105.4 |
| Double DQN, width 64 | 500.0 |

These are single-training-seed results with different configurations. Replay DQN experienced late instability. TD3's independent sections were verified in separate kernels and their outputs combined; Run All executes them sequentially.

Rendering is off by default for CartPole. Pass `render=True` for animation. The trading renderer is opt-in because it starts a blocking web server. Full notebook training can take tens of minutes on CPU.
