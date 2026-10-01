"""Build a separate PPO adaptation; never modify the original project notebook."""
from copy import deepcopy
import hashlib
import inspect
import json

import nbformat
import ppo_trading as p
import trading_tuning as t


def build():
    original_path = p.ROOT/'Project_Instruction.ipynb'
    original = nbformat.read(original_path,as_version=4)
    notebook = deepcopy(original)
    notebook.cells = []
    notebook.metadata['kernelspec'] = {'display_name':'RL Python 3.12','language':'python','name':'rl-project'}
    notebook.metadata['ppo_copy'] = dict(source=original_path.name,
        original_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256((p.ROOT/'ppo_trading.py').read_bytes()).hexdigest())
    def md(source): notebook.cells.append(nbformat.v4.new_markdown_cell(source.strip()))
    def code(source): notebook.cells.append(nbformat.v4.new_code_cell(source.strip()))

    md('''# Trading project — Proximal Policy Optimization (PPO)

A separate adaptation of `Project_Instruction.ipynb`. The original DQN notebook and course PDF remain unchanged. This version implements PPO in PyTorch, establishes a PPO baseline, tunes it using chronological validation, and compares the selected policy with DQN and market benchmarks.

**Run All** validates the expanded data, runs a short PPO training smoke test, and displays the completed tuning results when available. It reloads the selected checkpoint and independently verifies its evaluation return. Set `RETRAIN_ALL=True` in the experiment cell to train and evaluate the full search. Historical results in `eval/ppo/` belong to the previous data/action setup and are not results for this version. Select the **RL Python 3.12** kernel and run from the project folder. New PPO outputs live in `eval/ppo_202004_nine_actions/`.''')
    overview = deepcopy(original.cells[1])
    overview.source = overview.source.replace(
        'Training uses the preceding year, with no overlap.',
        'This PPO extension uses training history from April 1, 2020, with no evaluation overlap, '
        'as requested. This extends the original assignment’s one-year training rule.')
    notebook.cells.append(overview)
    md('''## 1. Environment and cached market data

The same dependencies as the DQN project are sufficient; no additional RL library is needed. Install `requirements.txt` once into Python 3.12. The code below contains the PPO implementation; `ppo_trading.py` exposes the identical implementation for repeatable multi-process experiments.

The expanded Binance spot BTC/USDT cache starts on March 25, 2020, providing seven days of feature warm-up before training starts on April 1, 2020. The evaluation window remains August 17–September 17, 2026 (end exclusive). Rebuild the cache with `.venv-rl/bin/python btc_data.py`; Run All uses the local cache without network access. Download provenance and checksums are stored beside it. The source omits 24 hourly candles. They remain absent: rolling features restart after each gap (with a fresh warm-up), and training samples only continuous hourly windows. No synthetic candles are inserted. See [Binance public-data documentation](https://github.com/binance/binance-public-data).''')
    code('''# %pip install -r requirements.txt
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib
import json
import inspect
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical
import gymnasium as gym
import gym_trading_env
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
import ppo_trading as experiment
from trading_tuning import ROOT, FOLDS, SEEDS, rollout
from btc_data import DATA_PATH, TRAIN_START, EVAL_START, EVAL_END, load_data
from ppo_trading import CONFIRM_STEPS, AGGREGATIONS, POSITIONS, config_from_dict
from trading_tuning import add_features as shared_add_features

torch.set_num_threads(1)
OUT = experiment.OUT
print('Python:', sys.version.split()[0], '| PyTorch:', torch.__version__)
print('PPO artifacts:', OUT)''')
    md('''### Causal features and strict date boundaries

The basic state contains the one-hour return, OHLC ratios, and relative volume. The momentum state adds returns over 1/6/24/72/168 hours, volatility, moving-average deviations, candle range/body, and relative volume. Two environment features describe the previous and current real position. Each training fold fits its own scaler; validation and evaluation never update it.

OHLCV candles are indexed by opening time. Their features become available at candle close; the simulator chooses exposure at that close for the next close-to-close return. This is an idealized execution assumption: trading fees and borrowing costs are included; spread, slippage, and latency are absent.''')
    code(inspect.getsource(p.add_features))
    code('''raw = load_data()
pre_evaluation = raw.loc[raw.index < EVAL_START]
features = add_features(pre_evaluation, 'momentum')
df_train = features.loc[features.index >= TRAIN_START].copy()
expected_training = pd.date_range(TRAIN_START, EVAL_START, freq='h', inclusive='left')
assert df_train.index.min() == TRAIN_START and df_train.index.max() == expected_training[-1]
assert df_train.index.is_unique and df_train.index.is_monotonic_increasing
print('Missing source hours:', len(raw.attrs.get('missing_hours', [])))
print('Additional warm-up rows excluded after gaps:', len(expected_training.difference(df_train.index)) - len(raw.attrs.get('missing_hours', [])))
print(f'{len(raw):,} cached candles; {len(df_train):,} training candles')
print('Training:', df_train.index.min(), 'through', df_train.index.max())
folds = pd.DataFrame([
    {'fold': i, 'training_start': TRAIN_START, 'training_end_exclusive': a,
     'validation_start': a, 'validation_end_exclusive': b,
     'training_hours': int(((df_train.index >= TRAIN_START) & (df_train.index < a)).sum()),
     'validation_hours': int(((df_train.index >= a) & (df_train.index < b)).sum())}
    for i, (a, b) in enumerate(FOLDS)
])
display(folds)
display(df_train.head(3))''')
    md('''### Trading environment

Every PPO candidate uses `[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]`. These are nine discrete target exposures: negative values short BTC, zero holds cash, and positive values hold BTC. The environment takes integer action indices 0–8; cash is index 4. The actor outputs nine categorical probabilities. This is a discrete spectrum, not a continuous action space. Every policy starts in cash with portfolio value 1,000. The fee parameter is 0.0001 and hourly borrowing-interest parameter is 0.000003. Evaluation covers the complete assigned history; training uses randomly located episodes of at most 720 candles.''')
    code(inspect.getsource(t.make_env))
    code('''example_env = make_env(df_train.iloc[:400], POSITIONS)
np.random.seed(23)
observation, info = example_env.reset(seed=23)
print('Observation shape:', observation.shape, '| Actions:', example_env.action_space)
next_observation, reward, terminated, truncated, info = example_env.step(POSITIONS.index(0.))  # Cash is index 4
print('Cash transition log reward:', reward)
example_env.close()''')
    md(r'''## 2. PPO implementation

PPO learns a **categorical actor** $\pi_\theta(a|s)$ and a **critic** $V_\phi(s)$. Stochastic action sampling provides training exploration; evaluation takes the most probable action. PPO suits this continuous-observation, discrete-position problem.

Fresh on-policy rollouts supply old action log probabilities and advantages. Several minibatch passes optimize the clipped objective

$$L^{\rm clip}=\mathbb{E}[\min(r_t\hat A_t,\;\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)],\qquad
r_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\rm old}(a_t|s_t)}.$$

The minimized loss is $-L^{\rm clip}+c_v\tfrac12(V-R)^2-c_e H(\pi)$. Advantages are normalized, gradients are clipped, and an approximate-KL threshold can stop an update early. Separate two-layer actor/critic networks use tanh activations and orthogonal initialization. Completed rollouts are discarded after updating.

Reference: [Schulman et al., *Proximal Policy Optimization Algorithms* (2017)](https://arxiv.org/abs/1707.06347).''')
    code(inspect.getsource(p.PPOConfig)+'\n\n'+inspect.getsource(p.ActorCritic))
    md(r'''### Generalized advantage estimation and episode boundaries

$$\delta_t=r_t+\gamma(1-\text{terminated}_t)V(s_{t+1})-V(s_t),\qquad
\hat A_t=\delta_t+\gamma\lambda(1-\text{done}_t)\hat A_{t+1}.$$

A time limit bootstraps from the **final observation before reset** but stops the advantage recursion across episodes. Bankruptcy is a true terminal and has zero bootstrap. This distinction also applies when a training rollout ends partway through an episode.''')
    code(inspect.getsource(p.compute_gae)+'\n\n'+inspect.getsource(p.clipped_policy_loss))
    code(inspect.getsource(p.PPOAgent))
    md('''### Training loop

Four environments sample independent, continuous hourly windows. Episodes never cross missing source candles or feature warm-up gaps. Feature normalization uses the supplied training frame only. Learning rewards are log portfolio returns multiplied by 100, with an optional penalty proportional to absolute position change. A bankruptcy penalty corrects the environment's zero terminal reward. **Reported returns use the actual unshaped portfolio value**, including the environment's costs.

This environment release uses NumPy's global generator for random starts, so every reset explicitly seeds it. PyTorch and minibatch sampling also receive the run's seed.''')
    code(inspect.getsource(p.TrainingWindowEnv)+'\n\n'+inspect.getsource(p.fit))
    md('''### Ensemble and checkpoint persistence

The search compares two predefined deterministic rules: choose the maximum mean action probability, or trade only when all three seeded policies agree (otherwise cash). The checkpoint stores the actor and critic, feature order, training scaler, configuration, seeds, and aggregation rule.''')
    code(inspect.getsource(p.PPOEnsemble))
    md('''## 3. Baseline and registered tuning experiment

The baseline uses basic features, learning rate 0.0003, discount 0.99, no entropy bonus, and a fixed learning rate. All candidates use the same causal data, reward scaling, costs, architecture, rollout size, and evaluation rules. Candidate changes are hypotheses to check, not assumed improvements.

1. Screen seven configurations on four earlier 30-day folds, using seed 23 and 16,384 transitions per fit.
2. Confirm the strongest two configurations plus the baseline using seeds 23, 42, and 71 and 49,152 transitions per fit.
3. Select the configuration and ensemble rule by `mean(log return) − 0.25 × mean(drawdown magnitude) − 0.10 × std(log return)` across the four folds.
4. Freeze the choice, refit on the complete history from April 1, 2020, and compare on the assignment month.

The assignment month was viewed during earlier DQN work. It is explicitly a **reused holdout**, and does not choose these PPO settings. The source/data fingerprints in `protocol.json` prevent mixing a changed implementation with cached trial results. Workers are ordinary independent Python training processes.''')
    code('''candidates = pd.DataFrame([asdict(c) for c in experiment.CONFIGS])
display(candidates[['name', 'features', 'positions', 'lr', 'gamma', 'gae_lambda',
                    'clip_range', 'entropy_coef', 'turnover_penalty', 'anneal_lr']])

RETRAIN_ALL = False
if RETRAIN_ALL:
    subprocess.run([sys.executable, str(ROOT / 'ppo_trading.py'), 'all', '--workers', '3'],
                   cwd=ROOT, check=True)
protocol = experiment.protocol()
assert not protocol['evaluation_used_for_selection']
if (OUT / 'screening_summary.csv').exists():
    display(pd.read_csv(OUT / 'screening_summary.csv').set_index('config').round(4))
else:
    print('Full experiment has not run for this data/action setup. Set RETRAIN_ALL=True to run it.')''')
    md('''### Screening results after retraining

Compare net return, drawdown, and position changes together. A lower training loss alone cannot select a trading policy. The screening ranking motivates the longer, multi-seed confirmation; the final ensemble may rank differently from individual short training runs. Baseline PPO receives the same confirmation and final-fit budgets as the selected candidate.''')
    code('''display(pd.read_csv(OUT / 'confirmation_individual_summary.csv').set_index('config').round(4))
display(pd.read_csv(OUT / 'ensemble_summary.csv').set_index('config').round(4))
selection = json.loads((OUT / 'selected_configuration.json').read_text())
print('Frozen selection:', selection['config']['name'], '/', selection['aggregation'])
display(pd.Series(selection['config'], name='Selected value').to_frame())
display(Image(filename=str(OUT / 'validation_comparison.png')))''')
    md('''### Training diagnostics and seed variation

Entropy measures policy concentration; value loss and explained variance describe the critic; approximate KL shows update size. These diagnostics help detect numerical or optimization problems. They do not measure out-of-sample profitability. Each trace below comes from one final training seed.''')
    code('''display(Image(filename=str(OUT / 'training_diagnostics.png')))
display(pd.read_csv(OUT / 'final_seed_results.csv')[
    ['policy','seed','portfolio_return','max_drawdown','action_changes']].round(4))''')
    md('''## 4. Reload and independently verify the final policy

The complete assignment window contains 744 candles and 743 decisions. The comparison includes baseline PPO, both existing DQN versions, cash, and fee-adjusted buy-and-hold. Each model uses its own saved training scaler and compatible feature set.''')
    code('''agent = PPOEnsemble.load(OUT / 'final_ppo_agent.pth')
all_features = add_features(load_data(), agent.config.features)
df_eval = all_features.loc[(all_features.index >= EVAL_START) & (all_features.index < EVAL_END)].copy()
assert len(df_eval) == 744
metrics, trace = rollout(agent, df_eval, return_trace=True)
saved_results = pd.read_csv(OUT / 'evaluation_results.csv')
assert len(saved_results) == 10 and saved_results.steps.eq(743).all()
np.testing.assert_allclose(saved_results.portfolio_return, metrics['portfolio_return'], rtol=0, atol=1e-12)
np.testing.assert_allclose(metrics['total_reward'], metrics['log_return'], rtol=0, atol=1e-10)
print('Reloaded policy return:', f"{metrics['portfolio_return']:.2%}")
display(saved_results.round(6))
comparison = pd.read_csv(OUT / 'policy_comparison.csv').set_index('policy')
display(comparison[['portfolio_return','max_drawdown','action_changes']].style.format(
    {'portfolio_return':'{:.2%}', 'max_drawdown':'{:.2%}', 'action_changes':'{:.0f}'}))
display(Image(filename=str(OUT / 'policy_comparison.png')))''')
    md('''Ten identical deterministic runs are reproducibility checks on one historical sequence. They are not ten independent observations. Drawdowns in raw tables are negative peak-relative returns; the written report displays their positive magnitudes. Position-change counts include the first entry from cash.''')
    md('''### Optional interactive environment renderer

The static plots above work on Run All. Enable the native renderer only when you want its local web interface; it starts a blocking server.''')
    code('''RUN_RENDERER = False
if RUN_RENDERER:
    from gym_trading_env.renderer import Renderer
    renderer = Renderer(render_logs_dir=str(OUT / 'render_logs'))
    renderer.run()''')
    md('''## 5. Measured results and interpretation''')
    code("display(Markdown((OUT / 'results_summary.md').read_text()))")
    md('''## 6. Presentation outline (10–15 minutes)

1. **Task and data (2 min):** hourly BTC/USDT, portfolio positions, fees, causal features, and chronological splits.
2. **PPO (3 min):** actor/critic, sampled actions, clipped probability ratios, GAE, and correct episode boundaries.
3. **Baseline and tuning (4 min):** identify the baseline's return, drawdown, and turnover; explain the candidate hypotheses and show screening versus multi-seed confirmation.
4. **Results (3 min):** compare the frozen PPO selection with baseline PPO, DQN, cash, and buy-and-hold. Show the portfolio trajectory and positions rather than training loss alone.
5. **Limits and next experiment (1–2 min):** discuss bearish validation bias, seed variability, missing execution costs, and the previously viewed evaluation month. A later untouched period or forward paper-trading study is needed to assess generalization.

Submission artifacts for this PPO version: **this notebook**, `ppo_trading.py`, `ppo_report.py`, `trading_tuning.py` (shared environment helpers), and **`eval/ppo_202004_nine_actions/evaluation_results.csv`**. Include the saved model and cache for full reproduction. The root-level DQN submission remains separate.''')
    md('''### Verification commands

```sh
.venv-rl/bin/python tests/test_ppo.py
.venv-rl/bin/python tests/test_tuning.py
.venv-rl/bin/python ppo_trading.py all --workers 3
```

Synthetic tests cover clipped gradients, terminal versus time-limit bootstrapping, actor and critic updates, incomplete rollouts, reproducible resets, frozen normalization, ensemble agreement, and checkpoint/backtest parity. Market performance uses only the real cached Binance history.''')
    for cell in notebook.cells:
        if cell.cell_type == 'code' and cell.source.startswith((
                "display(pd.read_csv(OUT / 'confirmation_individual_summary.csv')",
                "display(Image(filename=str(OUT / 'training_diagnostics.png')))",
                "agent = PPOEnsemble.load(OUT / 'final_ppo_agent.pth')",
                "display(Markdown((OUT / 'results_summary.md').read_text()))")):
            cell.source = ("if (OUT / 'results_summary.md').exists():\n" +
                           "\n".join('    ' + line for line in cell.source.splitlines()) +
                           "\nelse:\n    print('No new evaluation results yet; full retraining is required.')")
    # Run a bounded real-data integration check without representing it as a tuned model.
    notebook.cells.insert(19, nbformat.v4.new_markdown_cell(
        "### Expanded-history smoke test\n\n"
        "A 2,048-transition fit verifies the nine-action actor, training loop, and checkpoint reload. "
        "This is an integration check, not a performance experiment."))
    notebook.cells.insert(20, nbformat.v4.new_code_cell("""smoke_config = PPOConfig('nine_action_smoke')
smoke_agent, smoke_history = fit(df_train, smoke_config, seed=23, steps=2048)
assert smoke_agent.positions == POSITIONS
assert smoke_agent.network.actor[-1].out_features == 9
assert np.isfinite(smoke_history.to_numpy()).all()
OUT.mkdir(parents=True, exist_ok=True)
smoke_path = OUT / 'smoke_ppo_agent.pth'
PPOEnsemble([smoke_agent]).save(smoke_path)
smoke_loaded = PPOEnsemble.load(smoke_path)
assert smoke_loaded.positions == POSITIONS
print('Nine-action real-data training and checkpoint reload passed.')
display(smoke_history.tail())"""))
    path = p.ROOT/'Project_PPO.ipynb'
    nbformat.validate(notebook)
    nbformat.write(notebook,path)
    print(f'Built {path.name}: {len(notebook.cells)} cells')


if __name__ == '__main__':
    build()
