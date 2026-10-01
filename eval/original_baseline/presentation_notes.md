# Deep reinforcement learning for BTC/USDT trading

Speaking notes for a 10–15-minute class presentation. The figures and tables referenced below are saved in `Project_Instruction.ipynb` and `eval/`.

## 1. Task and main result — 1 minute

The task is to choose a portfolio position each hour using historical BTC/USDT data. I implemented DQN with experience replay and a target network, then compared feature standardization and Double DQN with the initial baseline.

The main finding is that the proposed modifications did not improve validation returns within this training budget. The baseline had the highest mean validation return, about 0.65%, but its retrained policy lost 20.37% in the final evaluation period while the market rose 21.12%. These results demonstrate the difference between a functioning reinforcement-learning implementation and a policy that generalizes to a new market period.

## 2. Data and environment — 1½ minutes

The dataset contains hourly Binance BTC/USDT candles. The permitted training year runs from August 17, 2025, to August 17, 2026, excluding the final endpoint. It contains 8,760 candles. The fixed evaluation period runs from August 17 to September 17, 2026, again excluding the endpoint. It contains 744 candles.

The notebook checks every expected timestamp, duplicate timestamps, ordering, finite values, and OHLCV consistency. It downloads seven extra days before training so the rolling volume feature has enough history. All rolling features use only information available at the current candle or earlier.

The state contains the close-price percentage change, open/high/low divided by close, and volume divided by its trailing seven-day maximum. The environment adds the previous position and the real current position.

The agent chooses among positions −1, 0, 0.5, 1, and 2. These represent a short position, cash, half exposure, full exposure, and leveraged long exposure. An action is the index of a position, not its numerical position value. The environment charges transaction and borrowing costs. Its reward is the logarithmic change in portfolio value after those costs.

Source: https://gym-trading-env.readthedocs.io/en/latest/environment_desc.html

## 3. Why DQN and how it works — 2 minutes

DQN fits this problem because the observations are continuous but the action set has only five choices. A neural network maps each observation to five estimated Q-values. A Q-value estimates the expected discounted future reward associated with an action.

During training, epsilon-greedy action selection initially explores frequently. Epsilon then declines toward 0.05. Evaluation always chooses the highest-valued action without exploration.

Each transition enters a replay buffer. Random mini-batches reduce dependence between consecutive observations and allow the algorithm to reuse experience. The buffer holds up to 50,000 transitions. Updates use batches of 64 after a 256-transition warm-up, with one optimizer update every four environment transitions.

The ordinary DQN target is the immediate reward plus gamma times the maximum next-state Q-value from the target network. Gamma is 0.99. A true terminal state removes the future-value term. A time-limit truncation ends the rollout but still permits bootstrapping.

The online network has two hidden layers of 64 units. Training uses Adam at learning rate 0.0001, Huber loss, and gradient clipping. A separate target network copies the online weights every 1,000 environment steps, equivalent to 250 scheduled optimizer updates once warm-up has ended.

## 4. Baseline evidence — 1½ minutes

Show `eval/training_comparison.png`, focusing first on the baseline curve.

The prediction loss becomes small, but this does not establish that the policy earns positive returns. The baseline's average training log return over the final ten episodes was negative for every seed: approximately −0.152, −0.103, and −0.083. Training still includes exploration, so those numbers are not equivalent to greedy evaluation performance.

On validation, the three baseline returns were about +2.01%, +5.06%, and −5.12%. Their mean was +0.65%, with a standard deviation of 5.22 percentage points. The variation is substantial relative to the mean. The market return over this validation window was approximately −1.58%, and remaining in cash provides a zero-return reference under this setup.

The baseline therefore provides a working comparison, but not strong evidence of a robust trading advantage.

## 5. Optimization hypotheses and fair comparisons — 2 minutes

The first proposed change was feature standardization. Raw return features are small, while ratios such as open divided by close are near one. Standardization could make optimization less sensitive to those differing scales. The means and standard deviations come only from the fitting period. Dynamic position features retain their natural scale.

The second proposed change was Double DQN. Ordinary DQN takes a maximum over noisy estimates. Double DQN uses the online network to select a next action, then uses the target network to evaluate that selected action. This separation can reduce maximization bias, but it does not guarantee better returns.

The experiment used three configurations and three matched seeds: 23, 42, and 71. Each trial ran 40 episodes, with at most 720 observations per episode. Architecture, positions, transaction costs, training budget, and other hyperparameters stayed fixed. Comparing scaled DQN with the raw baseline examines scaling. Comparing scaled Double DQN with scaled DQN examines the update rule conditional on scaling.

The final 30 days of the permitted training year serve as validation. The evaluation month is not used for model selection. After choosing the best configuration by mean validation return, the code retrains it on the entire permitted training year with the predeclared seed 23.

## 6. What worked and what did not — 1½ minutes

Show the table in `eval/results_summary.md`.

| Configuration | Mean validation return | Standard deviation across seeds | Mean maximum drawdown |
|---|---:|---:|---:|
| Baseline DQN | +0.65% | 5.22 pp | 5.66% |
| Scaled DQN | −6.44% | 7.74 pp | 14.31% |
| Scaled Double DQN | −9.22% | 6.64 pp | 12.97% |

Neither attempted change improved the baseline. Scaling reduced mean validation return by 7.09 percentage points. Scaled Double DQN reduced it by 9.87 percentage points relative to the raw baseline.

The scaled policies also changed actions more often: about 395 and 516 times on average, compared with 198 for the baseline. Higher turnover increases exposure to transaction costs. Greater sensitivity to noisy inputs is another plausible explanation. These are hypotheses because the experiment did not separately isolate their causal effects.

The evidence supports retaining the baseline for final evaluation rather than selecting the most complex method by assumption.

## 7. Final evaluation and limitations — 1½ minutes

The retrained baseline returned −20.37%, against a market return of +21.12%. Its excess return was −41.49 percentage points, and its maximum drawdown was 29.44%. It changed actions 16 times during this final period.

The evaluation CSV contains ten episodes. Each runs 743 transitions between the 744 hourly observations. The policy is deterministic, the data are identical, and every episode starts in cash. Therefore all ten results are identical. They demonstrate repeatability, not ten independent market tests.

The saved checkpoint includes both policy weights and preprocessing information. Reloading it reproduced the submission return. The implementation checks also verified replay updates, terminal masks, target-network independence and synchronization, and the different DQN and Double DQN target calculations.

The final result is poor despite selection on validation. Possible explanations include market-regime differences, weak predictive information in the features, model-selection noise, and the modest 40-episode training budget. This experiment cannot identify one explanation conclusively.

## 8. Further experiments and Q&A — 1 minute

Future work should use additional chronological validation windows and more training seeds before revisiting the final held-out period. Turnover penalties, smaller action sets, and longer training budgets are reasonable hypotheses to test. Each change should have a fixed budget and a clearly defined comparison. None of these proposed experiments has been performed in the recorded results.

Likely questions:

- **Why not select the best evaluation result?** That would use the held-out period for tuning and make the evaluation optimistic.
- **Why can a small loss accompany poor returns?** Prediction consistency under the training distribution is different from finding profitable actions in a new period.
- **Why did Double DQN perform worse?** The update rule addresses one source of bias. It does not solve noisy features, regime changes, limited data, or optimization sensitivity.
- **Why allow leveraged positions?** They are part of the provided action set. The environment models their financing costs, and drawdown reporting reveals part of their risk.
- **What does the failed result establish?** The code and comparison are reproducible. The tested methods did not produce a reliable trading advantage under these settings.
