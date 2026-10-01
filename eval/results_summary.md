# DRL tuning results

Selected configuration: **momentum_turnover**, with **unanimous** aggregation across seeds 23, 42, and 71.
The tuned policy returned **-6.04%**, versus **-20.37%** for the original policy: a **+14.33 percentage-point** change.

| Policy | Net return | Excess over market | Max drawdown | Position changes including entry |
|---|---:|---:|---:|---:|
| original_dqn | -20.37% | -41.49% | -29.44% | 17 |
| tuned_ensemble | -6.04% | -27.16% | -8.44% | 70 |
| cash | 0.00% | -21.12% | 0.00% | 0 |
| buy_and_hold | 21.11% | -0.01% | -7.71% | 1 |

## What changed

- Causal return, momentum, volatility, candle, and volume features with training-only normalization and clipping.
- Rewards multiplied by 100 for learning. Reported evaluation rewards and returns retain their original units.
- Double DQN with slowly updated targets, gradient clipping, and a shorter discount horizon.
- Allowed positions: [-1.0, 0.0, 0.5, 1.0, 2.0]. Additional training-only turnover penalty: 0.0003 × absolute position change.
- The final agreement rule requires all three networks to select the same position; disagreement returns to cash. Each member trains for 24,000 transitions.

## Selection evidence

Seven configurations were screened at 12,000 transitions on four expanding chronological training/validation folds. The strongest two were rechecked at 24,000 transitions with three seeds. Mean-Q ensembles remained weak, so a documented validation-only follow-up compared mean Q-values, majority vote, unanimous agreement, and normalized Q advantages. Selection used the validation score, never the assignment-period return.

| Finalist ensemble | Mean fold return | Worst fold return | Mean max drawdown | Selection score |
|---|---:|---:|---:|---:|
| momentum_turnover / unanimous | -0.88% | -1.79% | -4.50% | -0.0213 |
| momentum_turnover / normalized_advantage | -3.18% | -12.00% | -14.61% | -0.0769 |
| momentum_unlevered / unanimous | -3.51% | -15.45% | -11.20% | -0.0776 |
| momentum_turnover / mean_q | -4.01% | -5.98% | -15.79% | -0.0822 |
| momentum_turnover / majority | -4.64% | -14.85% | -13.76% | -0.0948 |
| momentum_unlevered / normalized_advantage | -4.73% | -16.02% | -15.87% | -0.0991 |
| momentum_unlevered / majority | -4.62% | -19.01% | -15.32% | -0.1006 |
| momentum_unlevered / mean_q | -7.42% | -16.95% | -15.94% | -0.1264 |

The fixed score is mean log return minus 0.25 × mean drawdown magnitude minus 0.10 × the standard deviation of log returns. Screening uses one seed; confirmation separately measures individual seeds and their ensemble. This is a limited search, not an exhaustive optimum.

Validation periods are 30-day windows ending every 60 days before 2026-08-17: Jan 19–Feb 18, Mar 20–Apr 19, May 19–Jun 18, and Jul 18–Aug 17. Every fold trains only on preceding observations. Three periods decline, so constant short and cash results are included in `validation_benchmarks.csv`.

## Interpretation and limits

The assignment period (2026-08-17 to 2026-09-17, UTC and end exclusive) was viewed during the first project run. This is therefore a **reused holdout comparison**, not a new independent test. The original tuning protocol and its agreement-rule amendment used only pre-evaluation validation. The selected policy was frozen before inspecting its new assignment-period result. An initial mean-Q final artifact was computed but not examined during agreement-rule selection and is retained in `initial_mean_q/`.
Ten evaluation episodes repeat the same deterministic policy and 744 candles, giving 743 transitions each. They are reproducibility checks, not independent observations. The original policy comparison now counts the first entry from cash, so turnover counts can differ by one from the earlier report.
The tuned policy did not beat the fee-adjusted buy-and-hold benchmark on this period. Improvement over the original policy alone does not establish a general trading advantage.

## Reproduce

```sh
.venv-rl/bin/python tests/test_tuning.py
.venv-rl/bin/python trading_tuning.py screen --workers 3
.venv-rl/bin/python trading_tuning.py confirm --workers 3
.venv-rl/bin/python trading_tuning.py aggregate
.venv-rl/bin/python trading_tuning.py final --workers 3
.venv-rl/bin/python tuning_report.py
```

The protocol records data and training-code fingerprints. Completed trials resume from their saved records. The original artifacts remain in `eval/original_baseline/`. Load the current policy using `ValidatedEnsemblePolicy.load("eval/final_trading_agent.pth")` from `trading_tuning`.
