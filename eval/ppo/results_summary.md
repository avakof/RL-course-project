# PPO trading project results

Selected **ppo_unlevered / unanimous**, combining training seeds 23, 42, and 71.
Tuned PPO returned **-0.19%**, a **+9.44 percentage-point** change versus baseline PPO and **+5.86 points** versus the existing tuned DQN.

| Policy | Net return | Maximum drawdown | Position changes |
|---|---:|---:|---:|
| Original DQN | -20.37% | 29.44% | 17 |
| Tuned DQN | -6.04% | 8.44% | 70 |
| Baseline PPO | -9.63% | 11.92% | 424 |
| Tuned PPO | -0.19% | 2.80% | 36 |
| Cash | 0.00% | 0.00% | 0 |
| Buy and hold | 21.11% | 7.71% | 1 |

## Implementation and tuning

The implementation uses a categorical actor and separate value critic, generalized advantage estimation, the clipped PPO objective, entropy regularization, gradient clipping, and a KL early-stop threshold. Rollouts are on-policy. Time limits bootstrap the final observation and break the advantage recursion at reset; bankruptcy does not bootstrap.
Initial baseline PPO averaged -14.04% per validation fold, with 313.5 position changes and 24.25% mean maximum drawdown. The strongest screening candidate, ppo_unlevered, averaged -0.97%, 118.2 changes, and 11.50% drawdown. This motivated the longer confirmation stage; it does not isolate the causal contribution of each changed hyperparameter.
Seven predefined configurations were screened for 16,384 transitions across four chronological folds. The best two and the baseline were confirmed for 49,152 transitions with three seeds. Mean-probability and unanimous-vote ensembles were registered before the search. The same fixed score selects the configuration and aggregation: mean log return minus 0.25 × mean drawdown magnitude minus 0.10 × log-return standard deviation.

| Confirmed policy | Mean fold return | Worst fold return | Mean drawdown | Score |
|---|---:|---:|---:|---:|
| ppo_unlevered::unanimous | 3.51% | 2.79% | 4.05% | 0.0237 |
| ppo_unlevered::mean_probability | 3.26% | -3.43% | 10.80% | -0.0028 |
| ppo_short_horizon::unanimous | -0.88% | -3.94% | 2.87% | -0.0184 |
| ppo_baseline::unanimous | -1.09% | -6.76% | 3.93% | -0.0286 |
| ppo_short_horizon::mean_probability | -8.10% | -13.13% | 14.33% | -0.1254 |
| ppo_baseline::mean_probability | -9.93% | -20.66% | 17.32% | -0.1606 |

| Validation benchmark | Mean fold return | Worst fold return | Mean drawdown |
|---|---:|---:|---:|
| Constant short | 9.31% | -7.86% | 9.09% |
| Cash | 0.00% | 0.00% | 0.00% |
| Buy and hold | -9.33% | -27.20% | 17.86% |

Final parameters: `{"anneal_lr": true, "batch_size": 256, "clip_range": 0.2, "entropy_coef": 0.01, "epochs": 6, "features": "momentum", "gae_lambda": 0.95, "gamma": 0.99, "hidden": 64, "lr": 0.0003, "max_grad_norm": 0.5, "n_envs": 4, "name": "ppo_unlevered", "positions": [-1.0, 0.0, 1.0], "reward_scale": 100.0, "rollout_steps": 256, "target_kl": 0.02, "turnover_penalty": 0.0001, "value_coef": 0.5}`.

## Evaluation and limitations

All policies use the same cached hourly BTC/USDT candles, 0.01% trading fee, and 0.000003 hourly borrowing-interest parameter. Features are causal; scalers fit only training data. Candles are indexed by opening time; decisions use the completed candle and simulate execution at its close before the next close-to-close movement. This assumes close-price execution and omits spread, slippage, and latency.
The fixed evaluation period is 2026-08-17 (inclusive) to 2026-09-17 (exclusive), UTC: 744 candles and 743 transitions. It was previously examined in the DQN project. This is a **reused holdout comparison**, not a fresh independent test. Its results did not select this PPO configuration. Ten identical deterministic episodes verify reproducibility; they are not independent samples and do not provide a confidence interval.
Validation windows end 180, 120, 60, and 0 days before evaluation. Three of the four windows decline; cash, constant-short, and buy-and-hold validation benchmarks reveal that directional imbalance. Model seed variation is saved separately, without choosing the best final seed.
The selected policy held cash for 91.25% of evaluation decisions. Much of the drawdown reduction comes from reduced market exposure; this also misses most of the rising market. The agreement rule controls participation and does not demonstrate stronger directional prediction.
Tuned PPO did not outperform fee-adjusted buy-and-hold during the reused evaluation period. Validation-selected improvements can fail to transfer to another market regime. These experiments do not establish a persistent trading advantage.

## Files and reproduction

- `Project_PPO.ipynb`: separate, executed copy adapted to PPO, with implementation, tables, and plots.
- `ppo_trading.py`: implementation and reproducible cached search.
- `eval/ppo/final_ppo_agent.pth`: selected ensemble with features, normalization, and aggregation.
- `eval/ppo/evaluation_results.csv`: ten reproducibility runs; `tuned_ppo_trace.csv`: full trajectory.
- `eval/ppo/protocol.json`: fixed candidates, folds, budgets, and source/data hashes.

```sh
.venv-rl/bin/python tests/test_ppo.py
.venv-rl/bin/python ppo_trading.py all --workers 3
.venv-rl/bin/python ppo_report.py
```

Completed matching trials are reused. To run a different protocol, preserve this experiment directory and use a new output directory. The existing DQN notebook, DQN artifacts, and course PDF are unchanged.

Algorithm reference: [Schulman et al., Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347).
