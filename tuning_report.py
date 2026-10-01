"""Publish the frozen tuning run and regenerate its notebook output section."""
from pathlib import Path
import ast
import contextlib
import io
import json
import shutil
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import trading_tuning as t


def load_original():
    notebook=nbformat.read(t.ROOT/'Project_Instruction.ipynb',as_version=4)
    ns={'__name__':'baseline_loader','POSITIONS':[-1,0,.5,1,2]}
    for c in notebook.cells:
        if c.cell_type!='code' or 'import trading_tuning' in c.source:
            continue
        nodes=[n for n in ast.parse(c.source).body if isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.FunctionDef))]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_notebook','exec'),ns)
    agent=ns['TradingDQN'].load(t.ROOT/'eval/original_baseline/final_trading_agent.pth')
    agent.positions=(-1.,0.,.5,1.,2.)
    return agent


def main():
    torch.set_num_threads(1)
    selected=json.loads((t.OUT/'selected_configuration.json').read_text())
    cfg=t.Config(**{**selected['config'],'positions':tuple(selected['config']['positions'])})
    raw=t.load_data()
    policy=t.ValidatedEnsemblePolicy.load(t.OUT/'tuned_trading_policy.pth')
    frame=t.add_features(raw,cfg.features)
    evaluation=frame.loc[(frame.index>=t.EVAL_START)&(frame.index<t.EVAL_END)]
    new_metrics,new_trace=t.rollout(policy,evaluation,return_trace=True)
    saved=pd.read_csv(t.OUT/'evaluation_results.csv')
    assert len(saved)==10 and saved.steps.eq(743).all()
    assert np.allclose(saved.portfolio_return,new_metrics['portfolio_return'],rtol=0,atol=1e-12)
    original=load_original()
    basic=t.add_features(raw,'basic')
    old_metrics,old_trace=t.rollout(original,basic.loc[(basic.index>=t.EVAL_START)&(basic.index<t.EVAL_END)],return_trace=True)
    cash_metrics,cash_trace=t.rollout(t.ConstantPolicy((0.,1.),0.),evaluation,return_trace=True)
    hold_metrics,hold_trace=t.rollout(t.ConstantPolicy((0.,1.),1.),evaluation,return_trace=True)
    comparison=pd.DataFrame([dict(policy=name,**m) for name,m in [
        ('original_dqn',old_metrics),('tuned_ensemble',new_metrics),('cash',cash_metrics),('buy_and_hold',hold_metrics)]])
    comparison.to_csv(t.OUT/'policy_comparison.csv',index=False)
    old_trace.to_csv(t.OUT/'original_policy_trace.csv',index=False)
    # Constant-position validation references expose directional market bias.
    validation_benchmarks=[]
    for fold,(start,end) in enumerate(t.FOLDS):
        part=frame.loc[(frame.index>=start)&(frame.index<end)]
        for position in [-1.,0.,1.]:
            m=t.rollout(t.ConstantPolicy((-1.,0.,1.),position),part)
            validation_benchmarks.append(dict(fold=fold,position=position,**m))
    pd.DataFrame(validation_benchmarks).to_csv(t.OUT/'validation_benchmarks.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for name,trace in [('Original DQN',old_trace),('Tuned ensemble',new_trace),('Buy and hold',hold_trace)]:
        values=trace.portfolio_value.to_numpy()
        axes[0].plot(pd.to_datetime(trace.timestamp),100*(values/1000-1),label=name)
        axes[1].plot(pd.to_datetime(trace.timestamp),100*(values/np.maximum.accumulate(values)-1),label=name)
    axes[0].set(title='Reused assignment period: cumulative return',ylabel='Return (%)')
    axes[1].set(title='Drawdown from portfolio peak',ylabel='Drawdown (%)')
    for ax in axes:
        ax.grid(alpha=.25);ax.legend();ax.tick_params(axis='x',rotation=25)
    fig.tight_layout();fig.savefig(t.OUT/'policy_comparison.png',dpi=150);plt.close(fig)
    rank=pd.read_csv(t.OUT/'aggregation_summary.csv').set_index(['config','aggregation'])
    screen=pd.read_csv(t.OUT/'screening_summary.csv',index_col='config')
    delta=new_metrics['portfolio_return']-old_metrics['portfolio_return']
    lines=['# DRL tuning results','',f'Selected configuration: **{cfg.name}**, with **{selected.get("aggregation","mean_q")}** aggregation across seeds 23, 42, and 71.',
           f'The tuned policy returned **{new_metrics["portfolio_return"]:.2%}**, versus **{old_metrics["portfolio_return"]:.2%}** for the original policy: a **{delta*100:+.2f} percentage-point** change.',
           '', '| Policy | Net return | Excess over market | Max drawdown | Position changes including entry |',
           '|---|---:|---:|---:|---:|']
    for _,r in comparison.iterrows():
        lines.append(f'| {r.policy} | {r.portfolio_return:.2%} | {r.excess_return:.2%} | {r.max_drawdown:.2%} | {int(r.action_changes)} |')
    lines+=['','## What changed','',
        '- Causal return, momentum, volatility, candle, and volume features with training-only normalization and clipping.',
        '- Rewards multiplied by 100 for learning. Reported evaluation rewards and returns retain their original units.',
        '- Double DQN with slowly updated targets, gradient clipping, and a shorter discount horizon.',
        f'- Allowed positions: {list(cfg.positions)}. Additional training-only turnover penalty: {cfg.turnover_penalty:g} × absolute position change.',
        '- The final agreement rule requires all three networks to select the same position; disagreement returns to cash. Each member trains for 24,000 transitions.',
        '', '## Selection evidence', '',
        'Seven configurations were screened at 12,000 transitions on four expanding chronological training/validation folds. The strongest two were rechecked at 24,000 transitions with three seeds. Mean-Q ensembles remained weak, so a documented validation-only follow-up compared mean Q-values, majority vote, unanimous agreement, and normalized Q advantages. Selection used the validation score, never the assignment-period return.',
        '', '| Finalist ensemble | Mean fold return | Worst fold return | Mean max drawdown | Selection score |',
        '|---|---:|---:|---:|---:|']
    for (name,aggregation),r in rank.iterrows():
        lines.append(f'| {name} / {aggregation} | {r.mean_return:.2%} | {r.worst_return:.2%} | {r.mean_drawdown:.2%} | {r.selection_score:.4f} |')
    lines+=['', 'The fixed score is mean log return minus 0.25 × mean drawdown magnitude minus 0.10 × the standard deviation of log returns. Screening uses one seed; confirmation separately measures individual seeds and their ensemble. This is a limited search, not an exhaustive optimum.',
        '', 'Validation periods are 30-day windows ending every 60 days before 2026-08-17: Jan 19–Feb 18, Mar 20–Apr 19, May 19–Jun 18, and Jul 18–Aug 17. Every fold trains only on preceding observations. Three periods decline, so constant short and cash results are included in `validation_benchmarks.csv`.',
        '', '## Interpretation and limits','',
        'The assignment period (2026-08-17 to 2026-09-17, UTC and end exclusive) was viewed during the first project run. This is therefore a **reused holdout comparison**, not a new independent test. The original tuning protocol and its agreement-rule amendment used only pre-evaluation validation. The selected policy was frozen before inspecting its new assignment-period result. An initial mean-Q final artifact was computed but not examined during agreement-rule selection and is retained in `initial_mean_q/`.',
        'Ten evaluation episodes repeat the same deterministic policy and 744 candles, giving 743 transitions each. They are reproducibility checks, not independent observations. The original policy comparison now counts the first entry from cash, so turnover counts can differ by one from the earlier report.',
        f'The tuned policy {"beat" if new_metrics["portfolio_return"]>hold_metrics["portfolio_return"] else "did not beat"} the fee-adjusted buy-and-hold benchmark on this period. Improvement over the original policy alone does not establish a general trading advantage.',
        '', '## Reproduce','',
        '```sh', '.venv-rl/bin/python tests/test_tuning.py',
        '.venv-rl/bin/python trading_tuning.py screen --workers 3',
        '.venv-rl/bin/python trading_tuning.py confirm --workers 3',
        '.venv-rl/bin/python trading_tuning.py aggregate',
        '.venv-rl/bin/python trading_tuning.py final --workers 3',
        '.venv-rl/bin/python tuning_report.py', '```','',
        'The protocol records data and training-code fingerprints. Completed trials resume from their saved records. The original artifacts remain in `eval/original_baseline/`. Load the current policy using `ValidatedEnsemblePolicy.load("eval/final_trading_agent.pth")` from `trading_tuning`.']
    report='\n'.join(lines)+'\n'
    (t.OUT/'results_summary.md').write_text(report)
    (t.ROOT/'eval/results_summary.md').write_text(report)
    for source,dest in [('evaluation_results.csv','evaluation_results.csv'),('tuned_trading_policy.pth','final_trading_agent.pth'),('selected_configuration.json','selected_configuration.json')]:
        shutil.copy2(t.OUT/source,t.ROOT/'eval'/dest)
    print(report)

if __name__=='__main__':main()
