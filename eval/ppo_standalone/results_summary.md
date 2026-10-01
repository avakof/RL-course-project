# Standalone PPO results

Selected **ppo_baseline::unanimous** using only pre-evaluation validation.

| Policy | Net return | Maximum drawdown | Position changes |
|---|---:|---:|---:|
| Baseline PPO | 14.44% | 4.02% | 434 |
| Selected PPO | -0.57% | 0.57% | 2 |
| Cash | 0.00% | 0.00% | 0 |
| Buy and hold | 21.11% | 7.71% | 1 |
| Original DQN | -20.37% | 29.44% | 17 |
| Tuned DQN | -6.04% | 8.44% | 70 |

The selected policy held cash for 99.87% of decisions.
Validation selection score: -0.000714; cash benchmark: 0.000000. Cash is not a selectable PPO configuration.
Selected minus baseline evaluation return: -15.01 percentage points.
Both PPO policies use the same three fitted networks. Only the aggregation rule differs; duplicated per-seed results are expected.

The evaluation month is a reused holdout. Ten repeated backtests are reproducibility checks, not independent samples. Trading fees and borrowing costs are included; spread, slippage, and latency are omitted. Training samples do not guarantee full coverage of the expanded history. Four recent validation windows do not represent all historical regimes.
The older DQN checkpoints are optional contextual benchmarks with different action spaces and training histories. No policy is replaced using its evaluation return.

Configuration: `{"anneal_lr": false, "batch_size": 256, "clip_range": 0.2, "entropy_coef": 0.0, "epochs": 6, "features": "basic", "gae_lambda": 0.95, "gamma": 0.99, "hidden": 64, "lr": 0.0003, "max_grad_norm": 0.5, "n_envs": 4, "name": "ppo_baseline", "positions": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0], "reward_scale": 100.0, "rollout_steps": 256, "target_kl": 0.02, "turnover_penalty": 0.0, "value_coef": 0.5}`
