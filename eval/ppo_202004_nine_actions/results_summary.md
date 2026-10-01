# PPO trading project results

Selected **ppo_baseline / unanimous**, combining training seeds 23, 42, and 71.
Training history starts 2020-04-01 UTC. All PPO candidates use the nine target exposures `[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]`. The older DQN policies retain their original training histories and action spaces, so they are contextual benchmarks rather than controlled PPO ablations.
Tuned PPO returned **-0.57%**, a **-15.01 percentage-point** change versus baseline PPO and **+5.47 points** versus the existing tuned DQN.

| Policy | Net return | Maximum drawdown | Position changes |
|---|---:|---:|---:|
| Original DQN | -20.37% | 29.44% | 17 |
| Tuned DQN | -6.04% | 8.44% | 70 |
| Baseline PPO | 14.44% | 4.02% | 434 |
| Tuned PPO | -0.57% | 0.57% | 2 |
| Cash | 0.00% | 0.00% | 0 |
| Buy and hold | 21.11% | 7.71% | 1 |

The selected policy and baseline use the same three fitted networks. The selected policy requires unanimous action agreement; the baseline takes the highest mean action probability. Selection changed the aggregation rule, not the winning network hyperparameters. Per-seed results are therefore identical between these two labels.

The selected validation score is -0.000714; the cash benchmark scores 0.000000. Cash is a reported benchmark, not a selectable PPO configuration. A top ranking among the trained policies does not establish an advantage over cash.

## Implementation and tuning

The implementation uses a categorical actor and separate value critic, generalized advantage estimation, the clipped PPO objective, entropy regularization, gradient clipping, and a KL early-stop threshold. Rollouts are on-policy. Time limits bootstrap the final observation and break the advantage recursion at reset; bankruptcy does not bootstrap.
Initial baseline PPO averaged -10.66% per validation fold, with 213.5 position changes and 17.11% mean maximum drawdown. The strongest screening candidate, ppo_momentum, averaged 1.37%, 188.0 changes, and 5.95% drawdown. This motivated the longer confirmation stage; it does not isolate the causal contribution of each changed hyperparameter.
Seven predefined configurations were screened for 16,384 transitions across four chronological folds. The best two and the baseline were confirmed for 49,152 transitions with three seeds. Mean-probability and unanimous-vote ensembles were registered before the search. The same fixed score selects the configuration and aggregation: mean log return minus 0.25 × mean drawdown magnitude minus 0.10 × log-return standard deviation.

| Confirmed policy | Mean fold return | Worst fold return | Mean drawdown | Score |
|---|---:|---:|---:|---:|
| ppo_baseline::unanimous | 0.04% | -0.24% | 0.31% | -0.0007 |
| ppo_low_turnover::unanimous | 0.22% | -1.31% | 1.74% | -0.0036 |
| ppo_momentum::unanimous | -0.73% | -2.56% | 1.87% | -0.0134 |
| ppo_low_turnover::mean_probability | 0.68% | -9.22% | 8.54% | -0.0265 |
| ppo_momentum::mean_probability | -1.46% | -11.07% | 9.43% | -0.0497 |
| ppo_baseline::mean_probability | -7.07% | -12.43% | 12.40% | -0.1099 |

| Validation benchmark | Mean fold return | Worst fold return | Mean drawdown |
|---|---:|---:|---:|
| Constant short | 9.31% | -7.86% | 9.09% |
| Cash | 0.00% | 0.00% | 0.00% |
| Buy and hold | -9.33% | -27.20% | 17.86% |

Final parameters: `{"anneal_lr": false, "batch_size": 256, "clip_range": 0.2, "entropy_coef": 0.0, "epochs": 6, "features": "basic", "gae_lambda": 0.95, "gamma": 0.99, "hidden": 64, "lr": 0.0003, "max_grad_norm": 0.5, "n_envs": 4, "name": "ppo_baseline", "positions": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0], "reward_scale": 100.0, "rollout_steps": 256, "target_kl": 0.02, "turnover_penalty": 0.0, "value_coef": 0.5}`.

## Evaluation and limitations

All policies use the same cached hourly BTC/USDT candles, 0.01% trading fee, and 0.000003 hourly borrowing-interest parameter. Features are causal; scalers fit only training data. Candles are indexed by opening time; decisions use the completed candle and simulate execution at its close before the next close-to-close movement. This assumes close-price execution and omits spread, slippage, and latency.
The fixed evaluation period is 2026-08-17 (inclusive) to 2026-09-17 (exclusive), UTC: 744 candles and 743 transitions. It was previously examined in the DQN project. This is a **reused holdout comparison**, not a fresh independent test. Its results did not select this PPO configuration. Ten identical deterministic episodes verify reproducibility; they are not independent samples and do not provide a confidence interval.
Validation windows end 180, 120, 60, and 0 days before evaluation. Three of the four windows decline; cash, constant-short, and buy-and-hold validation benchmarks reveal that directional imbalance. Model seed variation is saved separately, without choosing the best final seed.
The selected policy held cash for 99.87% of evaluation decisions, with mean absolute target exposure 0.001. These participation statistics describe the observed policy; they do not establish why its performance differs from the baseline.
The expanded history contains documented gaps. Features restart after each gap and training episodes sample only continuous hourly windows. The transition budget is unchanged: 49,152 transitions per confirmed or final fit. Random episodes do not guarantee that every historical candle is visited. Validation still covers four recent 2026 windows, not all historical market regimes.
Tuned PPO did not outperform fee-adjusted buy-and-hold during the reused evaluation period. Validation-selected improvements can fail to transfer to another market regime. These experiments do not establish a persistent trading advantage.

## Files and reproduction

- `Project_PPO.ipynb`: separate, executed copy adapted to PPO, with implementation, tables, and plots.
- `ppo_trading.py`: implementation and reproducible cached search.
- `eval/ppo_202004_nine_actions/final_ppo_agent.pth`: selected ensemble with features, normalization, and aggregation.
- `eval/ppo_202004_nine_actions/evaluation_results.csv`: ten reproducibility runs; `tuned_ppo_trace.csv`: full trajectory.
- `eval/ppo_202004_nine_actions/protocol.json`: fixed candidates, folds, budgets, and source/data hashes.

```sh
.venv-rl/bin/python tests/test_ppo.py
.venv-rl/bin/python ppo_trading.py all --workers 3
.venv-rl/bin/python ppo_report.py
```

Completed matching trials are reused. To run a different protocol, preserve this experiment directory and use a new output directory. The existing DQN notebook, DQN artifacts, and course PDF are unchanged.

Algorithm reference: [Schulman et al., Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347).
