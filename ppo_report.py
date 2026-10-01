"""Evaluate the frozen PPO selection and write only expanded-history PPO artifacts."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import ppo_trading as p
from trading_tuning import ValidatedEnsemblePolicy


def build_report():
    torch.set_num_threads(1)
    p.protocol()
    selected = json.loads((p.OUT/'selected_configuration.json').read_text())
    policy = p.PPOEnsemble.load(p.OUT/'final_ppo_agent.pth')
    baseline = p.PPOEnsemble.load(p.OUT/'baseline_ppo_agent.pth')
    if policy.config != p.config_from_dict(selected['config']) or policy.aggregation != selected['aggregation']:
        raise ValueError('PPO checkpoint differs from frozen selection')
    from tuning_report import load_original
    original = load_original()
    tuned_dqn = ValidatedEnsemblePolicy.load(p.ROOT/'eval/final_trading_agent.pth')
    raw = p.load_data()
    frames = {kind:p.add_features(raw,kind).loc[lambda x:(x.index>=p.EVAL_START)&(x.index<p.EVAL_END)]
              for kind in ['basic','momentum']}
    models = [('Original DQN',original,'basic'), ('Tuned DQN',tuned_dqn,tuned_dqn.config.features),
              ('Baseline PPO',baseline,baseline.config.features), ('Tuned PPO',policy,policy.config.features),
              ('Cash',p.ConstantPolicy((0.,1.),0.),'momentum'),
              ('Buy and hold',p.ConstantPolicy((0.,1.),1.),'momentum')]
    rows, traces = [], {}
    for name, model, kind in models:
        metrics, trace = p.rollout(model,frames[kind],return_trace=True)
        if metrics['steps'] != 743:
            raise AssertionError('Evaluation must cover all 743 assignment transitions')
        rows.append(dict(policy=name,**metrics))
        traces[name] = trace
        trace.to_csv(p.OUT/(name.lower().replace(' ','_')+'_trace.csv'),index=False)
    comparison = pd.DataFrame(rows).set_index('policy')
    comparison.to_csv(p.OUT/'policy_comparison.csv')
    metrics = comparison.loc['Tuned PPO'].to_dict()
    repeated = []
    for episode in range(10):
        result = p.rollout(policy,frames[policy.config.features])
        np.testing.assert_allclose(result['portfolio_return'],metrics['portfolio_return'],rtol=0,atol=1e-12)
        repeated.append(dict(episode=episode+1,**result))
    pd.DataFrame(repeated).to_csv(p.OUT/'evaluation_results.csv',index=False)
    seed_rows = []
    for name,model in [('Baseline PPO',baseline),('Tuned PPO',policy)]:
        for member in model.agents:
            seed_rows.append(dict(policy=name,seed=member.seed,
                **p.rollout(member,frames[member.config.features])))
    pd.DataFrame(seed_rows).to_csv(p.OUT/'final_seed_results.csv',index=False)
    # Native environment render history is optional to view, but saved here.
    p.rollout(policy,frames[policy.config.features],render_dir=p.OUT/'render_logs')

    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False})
    colors = {'Original DQN':'#999999','Tuned DQN':'#b77d33','Baseline PPO':'#7a60aa',
              'Tuned PPO':'#1667b1','Buy and hold':'#25916b','Cash':'#777777'}
    fig,axes = plt.subplots(2,2,figsize=(14,9))
    for name,trace in traces.items():
        if name == 'Cash': continue
        values = trace.portfolio_value.to_numpy()
        axes[0,0].plot(trace.timestamp,100*(values/1000-1),label=name,color=colors[name],
                       linewidth=2 if name=='Tuned PPO' else 1.2)
        if name in ['Tuned PPO','Tuned DQN','Buy and hold']:
            axes[0,1].plot(trace.timestamp,100*(values/np.maximum.accumulate(values)-1),
                          label=name,color=colors[name])
    axes[0,0].axhline(0,color='#777777',lw=.7,ls='--')
    axes[0,0].set(title='Cumulative net return',ylabel='Return (%)')
    axes[0,1].set(title='Drawdown from portfolio peak',ylabel='Drawdown (%)')
    chosen_trace = traces['Tuned PPO']
    axes[1,0].step(chosen_trace.timestamp,chosen_trace.position,where='post',color=colors['Tuned PPO'])
    axes[1,0].set(title='Tuned PPO: chosen exposure',ylabel='Position / portfolio value',
                  yticks=list(policy.positions))
    counts=chosen_trace.position.iloc[1:].value_counts().reindex(policy.positions,fill_value=0)
    axes[1,1].bar([str(x) for x in policy.positions],100*counts/counts.sum(),color=colors['Tuned PPO'])
    axes[1,1].set(title='Tuned PPO: exposure distribution',ylabel='Decisions (%)',xlabel='Position')
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.2)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.tick_params(axis='x',rotation=25)
    axes[0,0].legend(fontsize=9); axes[0,1].legend(fontsize=9)
    fig.suptitle('BTC/USDT · Reused assignment period · 17 Aug–17 Sep 2026 UTC',fontsize=14)
    fig.tight_layout(); fig.savefig(p.OUT/'policy_comparison.png',dpi=150); plt.close(fig)

    validation = pd.read_csv(p.OUT/'ensemble_validation.csv')
    benchmarks = pd.read_csv(p.OUT/'validation_benchmarks.csv')
    chosen_id = policy.config.name+'::'+policy.aggregation
    baseline_id = 'ppo_baseline::mean_probability'
    fold_table = pd.concat([validation.loc[validation.config.isin([chosen_id,baseline_id])],benchmarks])
    fold_table = fold_table.pivot(index='fold',columns='config',values='portfolio_return')
    fold_table = fold_table.rename(columns={chosen_id:'Tuned PPO',baseline_id:'Baseline PPO'})
    fold_table.to_csv(p.OUT/'validation_fold_comparison.csv')
    ax = (100*fold_table).plot.bar(figsize=(12,4.5),width=.8)
    ax.set(title='Pre-assignment validation: each fold trains only on earlier data',
           ylabel='Net return (%)',xlabel='30-day validation window')
    ax.set_xticklabels([f'{a:%d %b}–{b:%d %b}' for a,b in p.FOLDS],rotation=0)
    ax.axhline(0,color='#777777',lw=.8); ax.grid(axis='y',alpha=.2)
    ax.legend(fontsize=8,ncol=3)
    ax.figure.tight_layout(); ax.figure.savefig(p.OUT/'validation_comparison.png',dpi=150); plt.close(ax.figure)

    fig,axes = plt.subplots(2,2,figsize=(12,7))
    for seed in p.SEEDS:
        history = pd.read_csv(p.OUT/'final_members'/f'{policy.config.name}_fold-1_seed{seed}.training.csv')
        for ax,metric,title in zip(axes.flat,['entropy','value_loss','approx_kl','explained_variance'],
                                   ['Policy entropy','Value loss (scaled rewards)','Approximate KL divergence','Value explained variance']):
            ax.plot(history.steps,history[metric],label=f'Seed {seed}')
            ax.set(title=title,xlabel='Training transitions'); ax.grid(alpha=.2)
    axes[0,0].legend(); fig.tight_layout()
    fig.savefig(p.OUT/'training_diagnostics.png',dpi=150); plt.close(fig)

    ranked = pd.read_csv(p.OUT/'ensemble_summary.csv')
    screening = pd.read_csv(p.OUT/'screening_summary.csv').set_index('config')
    screening_baseline = screening.loc['ppo_baseline']
    screening_best = screening.iloc[0]
    improvement = 100*(metrics['portfolio_return']-comparison.loc['Baseline PPO','portfolio_return'])
    relative_dqn = 100*(metrics['portfolio_return']-comparison.loc['Tuned DQN','portfolio_return'])
    lines = ['# PPO trading project results','',
        f'Selected **{policy.config.name} / {policy.aggregation}**, combining training seeds 23, 42, and 71.',
        f'Training history starts {p.TRAIN_START:%Y-%m-%d} UTC. All PPO candidates use the nine target exposures `{list(p.POSITIONS)}`. The older DQN policies retain their original training histories and action spaces, so they are contextual benchmarks rather than controlled PPO ablations.',
        f'Tuned PPO returned **{metrics["portfolio_return"]:.2%}**, a **{improvement:+.2f} percentage-point** change versus baseline PPO and **{relative_dqn:+.2f} points** versus the existing tuned DQN.',
        '', '| Policy | Net return | Maximum drawdown | Position changes |', '|---|---:|---:|---:|']
    for name,row in comparison.iterrows():
        lines.append(f'| {name} | {row.portfolio_return:.2%} | {abs(row.max_drawdown):.2%} | {int(row.action_changes)} |')
    if policy.config == baseline.config:
        lines += ['', 'The selected policy and baseline use the same three fitted networks. The selected policy requires unanimous action agreement; the baseline takes the highest mean action probability. Selection changed the aggregation rule, not the winning network hyperparameters. Per-seed results are therefore identical between these two labels.']
    selected_score = float(ranked.loc[ranked.config == chosen_id, 'selection_score'].iloc[0])
    cash_score = float(p.ranking(benchmarks).loc['Cash', 'selection_score'])
    lines += ['', f'The selected validation score is {selected_score:.6f}; the cash benchmark scores {cash_score:.6f}. Cash is a reported benchmark, not a selectable PPO configuration. A top ranking among the trained policies does not establish an advantage over cash.']
    lines += ['', '## Implementation and tuning', '',
        'The implementation uses a categorical actor and separate value critic, generalized advantage estimation, the clipped PPO objective, entropy regularization, gradient clipping, and a KL early-stop threshold. Rollouts are on-policy. Time limits bootstrap the final observation and break the advantage recursion at reset; bankruptcy does not bootstrap.',
        f'Initial baseline PPO averaged {screening_baseline.mean_return:.2%} per validation fold, with {screening_baseline.mean_action_changes:.1f} position changes and {abs(screening_baseline.mean_drawdown):.2%} mean maximum drawdown. The strongest screening candidate, {screening.index[0]}, averaged {screening_best.mean_return:.2%}, {screening_best.mean_action_changes:.1f} changes, and {abs(screening_best.mean_drawdown):.2%} drawdown. This motivated the longer confirmation stage; it does not isolate the causal contribution of each changed hyperparameter.',
        'Seven predefined configurations were screened for 16,384 transitions across four chronological folds. The best two and the baseline were confirmed for 49,152 transitions with three seeds. Mean-probability and unanimous-vote ensembles were registered before the search. The same fixed score selects the configuration and aggregation: mean log return minus 0.25 × mean drawdown magnitude minus 0.10 × log-return standard deviation.',
        '', '| Confirmed policy | Mean fold return | Worst fold return | Mean drawdown | Score |',
        '|---|---:|---:|---:|---:|']
    for _,row in ranked.iterrows():
        lines.append(f'| {row.config} | {row.mean_return:.2%} | {row.worst_return:.2%} | {-row.mean_drawdown:.2%} | {row.selection_score:.4f} |')
    lines += ['', '| Validation benchmark | Mean fold return | Worst fold return | Mean drawdown |',
              '|---|---:|---:|---:|']
    for name,row in p.ranking(benchmarks).iterrows():
        lines.append(f'| {name} | {row.mean_return:.2%} | {row.worst_return:.2%} | {abs(row.mean_drawdown):.2%} |')
    lines += ['', f'Final parameters: `{json.dumps(selected["config"],sort_keys=True)}`.',
        '', '## Evaluation and limitations', '',
        'All policies use the same cached hourly BTC/USDT candles, 0.01% trading fee, and 0.000003 hourly borrowing-interest parameter. Features are causal; scalers fit only training data. Candles are indexed by opening time; decisions use the completed candle and simulate execution at its close before the next close-to-close movement. This assumes close-price execution and omits spread, slippage, and latency.',
        'The fixed evaluation period is 2026-08-17 (inclusive) to 2026-09-17 (exclusive), UTC: 744 candles and 743 transitions. It was previously examined in the DQN project. This is a **reused holdout comparison**, not a fresh independent test. Its results did not select this PPO configuration. Ten identical deterministic episodes verify reproducibility; they are not independent samples and do not provide a confidence interval.',
        'Validation windows end 180, 120, 60, and 0 days before evaluation. Three of the four windows decline; cash, constant-short, and buy-and-hold validation benchmarks reveal that directional imbalance. Model seed variation is saved separately, without choosing the best final seed.',
        f'The selected policy held cash for {(chosen_trace.position.iloc[1:]==0).mean():.2%} of evaluation decisions, with mean absolute target exposure {chosen_trace.position.iloc[1:].abs().mean():.3f}. These participation statistics describe the observed policy; they do not establish why its performance differs from the baseline.',
        'The expanded history contains documented gaps. Features restart after each gap and training episodes sample only continuous hourly windows. The transition budget is unchanged: 49,152 transitions per confirmed or final fit. Random episodes do not guarantee that every historical candle is visited. Validation still covers four recent 2026 windows, not all historical market regimes.',
        f'Tuned PPO {"outperformed" if metrics["portfolio_return"]>comparison.loc["Buy and hold","portfolio_return"] else "did not outperform"} fee-adjusted buy-and-hold during the reused evaluation period. Validation-selected improvements can fail to transfer to another market regime. These experiments do not establish a persistent trading advantage.',
        '', '## Files and reproduction', '',
        '- `Project_PPO.ipynb`: separate, executed copy adapted to PPO, with implementation, tables, and plots.',
        '- `ppo_trading.py`: implementation and reproducible cached search.',
        '- `eval/ppo_202004_nine_actions/final_ppo_agent.pth`: selected ensemble with features, normalization, and aggregation.',
        '- `eval/ppo_202004_nine_actions/evaluation_results.csv`: ten reproducibility runs; `tuned_ppo_trace.csv`: full trajectory.',
        '- `eval/ppo_202004_nine_actions/protocol.json`: fixed candidates, folds, budgets, and source/data hashes.',
        '', '```sh', '.venv-rl/bin/python tests/test_ppo.py',
        '.venv-rl/bin/python ppo_trading.py all --workers 3',
        '.venv-rl/bin/python ppo_report.py', '```', '',
        'Completed matching trials are reused. To run a different protocol, preserve this experiment directory and use a new output directory. The existing DQN notebook, DQN artifacts, and course PDF are unchanged.',
        '', 'Algorithm reference: [Schulman et al., Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347).']
    report='\n'.join(lines)+'\n'
    (p.OUT/'results_summary.md').write_text(report)
    print(comparison[['portfolio_return','max_drawdown','action_changes']].to_string(),flush=True)
    return comparison


if __name__ == '__main__':
    build_report()
