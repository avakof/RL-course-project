"""Synthetic data are used only for isolated implementation checks, never submission."""
import ast
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

n=json.loads(Path('Project_Instruction.ipynb').read_text())
ns={'__name__':'verification','POSITIONS':[-1,0,0.5,1,2]}
for cell in n['cells']:
    if cell['cell_type']!='code': continue
    tree=ast.parse(''.join(cell['source']))
    nodes=[x for x in tree.body if isinstance(x,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'Project_Instruction.ipynb','exec'),ns)
ns['torch'].set_num_threads(1)
rng=np.random.default_rng(10)
close=100*np.exp(np.cumsum(rng.normal(0,0.001,1000)))
frame=pd.DataFrame({'open':close,'high':close,'low':close,'close':close,'volume':np.ones(1000)},
                   index=pd.date_range('2020-01-01',periods=1000,freq='h',tz='UTC'))
frame['feature_return']=frame.close.pct_change().fillna(0)
agent,history=ns['fit_agent'](frame,normalize=True,double=True,episodes=1)
assert len(history)==1 and np.isfinite(history[['reward','loss']]).all().all()
assert agent.steps>256
metrics=ns['score_agent'](agent,frame)
assert all(np.isfinite(v) for v in metrics.values())
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'agent.pth'
    agent.save(path)
    restored=ns['TradingDQN'].load(path)
    state=np.array([0.001,0,0],dtype=np.float32)
    assert restored.choose_action_eval(state)==agent.choose_action_eval(state)
    assert np.array_equal(restored.mean,agent.mean)
    env=ns['make_env'](frame)
    try:
        out=ns['evaluate_agent'](restored,env,num_episodes=2,csv_path=Path(d)/'test_only.csv')
    finally: env.close()
    assert len(out)==2 and out.steps.eq(999).all()
    assert out.portfolio_return.nunique()==1
print('PASS: trading replay updates, scaling, checkpoint reload, wrapper metrics, full-window deterministic evaluation')
