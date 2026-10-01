### Observed results from the completed run

| Configuration | Mean validation return | Seed standard deviation | Mean max drawdown | Mean action changes |
|---|---:|---:|---:|---:|
| baseline_dqn | 0.65% | 5.22% | -5.66% | 198.0 |
| scaled_double_dqn | -9.22% | 6.64% | -12.97% | 515.7 |
| scaled_dqn | -6.44% | 7.74% | -14.31% | 394.7 |

Selected configuration: **baseline_dqn**. After retraining, final-period portfolio return was **-20.37%**, versus market return **21.12%**, for excess return **-41.49%**. Each evaluation episode ran 743 transitions across the 744 hourly candles.

The baseline last-ten-episode mean training log returns were negative for all three seeds despite small TD losses. A small regression loss therefore did not establish a profitable policy. Feature scaling addresses disparate input scales, while Double DQN addresses maximization bias; the table measures their actual effects.
scaled_dqn reduced mean validation return by 7.09% relative to baseline (percentage-point difference).
scaled_double_dqn reduced mean validation return by 9.87% relative to baseline (percentage-point difference).

The best validation score does not establish statistical significance with only three training seeds and one market period. The final ten deterministic repeats are identical and should not be treated as independent performance samples. A 40-episode training budget also limits the conclusions.

Final-period maximum drawdown was 29.44%, with 16 action changes. Reloading the saved policy reproduced the submitted return.
