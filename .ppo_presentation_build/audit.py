import json,hashlib,ast
from pathlib import Path
import numpy as np,pandas as pd
import gymnasium as gym
import gym_trading_env
root=Path('/Users/alivakof/Desktop/RL/PPO_Presentation')
j=json.loads((root/'data/verified_experiment.json').read_text())
def df(x):return pd.DataFrame(x['data'],index=x['index'],columns=x['columns'])
tables={k:df(v) for k,v in j['tables'].items()}
for k,t in tables.items():t.to_csv(root/'data'/f'{k}.csv',index=True)
traces={k:df(v) for k,v in j['traces'].items()}
for k,t in traces.items():t.to_csv(root/'data'/f"{k.lower().replace(' ','_')}_trace.csv",index=False)
for k,h in j['histories'].items():df(h).to_csv(root/'data'/f"training_seed_{ast.literal_eval(k)[2]}.csv",index=False)
comp=tables['policy_comparison'];b=comp.loc['Baseline PPO'];s=comp.loc['Selected PPO']
assert j['same_member_weights']
assert j['evaluation_index']['rows']==744 and j['evaluation_index']['continuous']
assert sum(len(tables[k+'_trials']) for k in ['screening','confirmation','final_members'])==67
assert sum(tables[k+'_trials'].training_steps.sum() for k in ['screening','confirmation','final_members'])==2375680
for k,tr in traces.items():
 vals=tr.portfolio_value.to_numpy();assert len(vals)==744
 np.testing.assert_allclose(vals[-1]/vals[0]-1,comp.loc[k,'portfolio_return'],atol=1e-12)
 np.testing.assert_allclose((vals/np.maximum.accumulate(vals)-1).min(),comp.loc[k,'max_drawdown'],atol=1e-12)
 np.testing.assert_allclose(np.expm1(tr.reward.sum()),comp.loc[k,'portfolio_return'],atol=1e-12)
 idx=pd.to_datetime(tr.timestamp,utc=True)
 assert idx.iloc[0]==pd.Timestamp('2026-08-17',tz='UTC') and idx.iloc[-1]==pd.Timestamp('2026-09-16 23:00',tz='UTC')
# Confirm the installed simulator's financing behavior on constant prices.
idx=pd.date_range('2026-01-01',periods=11,freq='h')
fr=pd.DataFrame({'open':100.,'high':100.,'low':100.,'close':100.,'volume':1.,'feature_constant':0.},index=idx)
e=gym.make('TradingEnv',df=fr,positions=[-1.,0.],initial_position=0.,trading_fees=0.,borrow_interest_rate=.000003,verbose=0)
e.reset();v=[]
for _ in range(10):
 _,_,_,_,info=e.step(0);v.append(float(info['portfolio_valuation']))
e.close()
assert np.allclose(v,[999.997]*10,rtol=0,atol=1e-9)
source=Path('/Users/alivakof/Desktop/RL/Project_PPO_All_In_One_Fixed.ipynb')
audit={'source_notebook':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'verified_at_utc':str(pd.Timestamp.now(tz='UTC')),
 'checks':{'recomputed_four_policy_paths':True,'exact_trace_metrics':True,'same_three_networks':True,'67_fits_2375680_transitions':True,'744_hours_743_transitions':True},
 'baseline_return_pct':100*b.portfolio_return,'selected_return_pct':100*s.portfolio_return,
 'selected_minus_baseline_pp':100*(s.portfolio_return-b.portfolio_return),
 'drawdown_reduction_pp':100*(abs(b.max_drawdown)-abs(s.max_drawdown)),
 'drawdown_reduction_relative_pct':100*(1-abs(s.max_drawdown)/abs(b.max_drawdown)),
 'changes_reduction_relative_pct':100*(1-s.action_changes/b.action_changes),
 'cash_fraction_pct':100*traces['Selected PPO'].position.iloc[1:].eq(0).mean(),
 'selected_nonzero_rows':json.loads(traces['Selected PPO'].loc[traces['Selected PPO'].position.ne(0)].to_json(orient='records')),
 'constant_price_financing_test':{'steps':10,'initial_value':1000,'rate_per_step':.000003,'observed_values':v,'expected_value_with_linear_accumulation':999.970,'interpretation':'The interest liability is overwritten each step rather than accumulated for a held short.'}}
(root/'data/audit.json').write_text(json.dumps(audit,indent=2))
(root/'data/simulator_audit.txt').write_text('\n\n'.join(k+'\n'+v for k,v in j['simulator_source'].items()))
print(json.dumps(audit,indent=2))
