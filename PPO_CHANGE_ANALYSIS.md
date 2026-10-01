# PPO notebook analysis and implemented changes

## Phase 1 — Proposed change

Extend hourly Binance spot BTC/USDT training to April 1, 2020 UTC, retain the fixed evaluation dates, and use nine categorical target exposures: `[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]`.

The notebook implements PPO with separate actor/critic networks, clipped policy loss, generalized advantage estimation, training-only feature normalization, chronological validation, and saved ensemble policies. This architecture already supports any finite position list; the actor output dimension is derived from that list.

## Phase 2 — Audit findings

- The previous loader required exactly the old cache dates, and the notebook asserted exactly 8,760 training candles. Merely adding downloaded files would not extend training.
- PPO candidates used several incompatible action lists. All seven now use the requested nine exposures; the former restricted-action candidates have been replaced by low-turnover and higher-entropy candidates.
- Cash changes from index 1 to index 4. Actions are integer indices, while positions are exposure values. This remains a discrete action space.
- Old checkpoints and reported returns belong to the old experiment. New artifacts use `eval/ppo_202004_nine_actions/`; result cells do not present old results as new measurements.
- Binance spot archives change timestamp units from milliseconds to microseconds in 2025. The downloader handles both and verifies archive checksums. Source: [Binance public-data documentation](https://github.com/binance/binance-public-data).
- The source omits 24 hourly candles. No prices or volumes were fabricated. Features restart after gaps, and sampled training episodes contain only consecutive hours.
- The original evaluation month was previously inspected. It remains a reused holdout. Fees and borrowing costs are modeled, but spread, slippage, and execution latency are not. More history does not establish improved returns.
- Validation dates and training transition budgets remain unchanged. The validation folds still emphasize recent 2026 conditions even though training now includes earlier market regimes.

## Phase 3 — Verification

1. **Does the downloaded history cover April 2020 and match the existing cache?** Yes: 66 monthly archives passed SHA-256 checks; 528 overlapping candles matched. The combined cache contains 56,784 actual candles from March 25, 2020 through September 16, 2026 at 23:00 UTC, including seven days of initial warm-up. All 24 missing timestamps are recorded in the cache metadata and JSON manifest.
2. **Do all requested actions execute correctly?** Yes: tests exercise all nine indices and verify resulting exposures. Cash is index 4 and produces zero reward when starting in cash. Actor probabilities have nine entries; checkpoint round trips preserve behavior.
3. **Does the notebook run with causal features and continuous training episodes?** Yes: all 17 code cells executed successfully, including a 2,048-transition real-data fit and checkpoint reload. Eleven PPO tests, two downloader tests, and eleven shared trading tests passed. Gap tests verify feature warm-up restarts and continuous sampled windows.

The momentum training frame contains 53,991 usable candles from April 1, 2020 through August 16, 2026. Relative to the complete calendar, 24 missing source candles and 1,881 additional post-gap warm-up candles are excluded. Evaluation retains all 744 candles and 743 transitions.

## Phase 4 — Implemented result

`Project_PPO.ipynb`, its generator, PPO implementation, report paths, and README now use the expanded-history setup. `btc_data.py` reproducibly rebuilds the cache from retained verified archives plus the existing newer cache. The DQN implementation, original notebook, and historical artifacts remain unchanged.

Full screening, confirmation, final training, and performance evaluation are now **complete**: 67 fits and 2,375,680 training transitions. Validation selected `ppo_baseline` with unanimous voting. Evaluation returned **−0.57%**, versus **+14.44%** for the same networks with mean-probability aggregation and **+21.11%** for buy-and-hold. The selected policy held cash for 742 of 743 decisions. Its lower drawdown did not translate into higher returns. See `eval/ppo_202004_nine_actions/results_summary.md` for the complete results.

To reproduce or resume the completed experiment:

```sh
.venv-rl/bin/python ppo_trading.py all --workers 3
```

Alternatively set `RETRAIN_ALL=True` in the notebook. The default notebook run performs the bounded integration check, displays the completed results, and independently verifies the final checkpoint backtest.

Delta from the proposal: explicit gap handling, timestamp-unit normalization, cash-index correction, and separation from incompatible old model artifacts were necessary in addition to changing the start date and exposure list.
