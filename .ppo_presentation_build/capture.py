from pathlib import Path
from jupyter_client import BlockingKernelClient
import time
root=Path('/Users/alivakof/Desktop/RL/PPO_Presentation')
code=r'''
def _capture_ppo_presentation():
    import json, inspect, hashlib, sys
    from pathlib import Path
    from importlib.metadata import version
    def table(df):
        return json.loads(df.to_json(orient='split', date_format='iso', double_precision=15))
    assert experiment_protocol(raw) == experiment['protocol']
    out = {'protocol':experiment['protocol'], 'selection':experiment['selection'],
           'verification':experiment['verification'], 'report':experiment['report'],
           'tables':{k:table(v) for k,v in experiment['tables'].items()},
           'traces':{k:table(v) for k,v in experiment['traces'].items()},
           'histories':{str(k):table(v['history']) for k,v in experiment['trials']['final_members'].items()},
           'raw':{'rows':len(raw),'start':str(raw.index.min()),'end':str(raw.index.max()),'attrs':raw.attrs},
           'daily_close':table(raw.close.resample('D').last().dropna().to_frame()),
           'versions':{p:version(p) for p in ['pandas','numpy','torch','gymnasium','gym-trading-env']},
           'feature_rows':{k:len(v.loc[v.index>=TRAIN_START]) for k,v in experiment['feature_frames'].items()}}
    out['recomputed']={}
    for name,agent in [('Baseline PPO',experiment['baseline']),('Selected PPO',experiment['policy']),('Cash',ConstantPolicy((0.,1.),0.)),('Buy and hold',ConstantPolicy((0.,1.),1.))]:
        kind = agent.config.features if hasattr(agent,'config') else 'momentum'
        fr=add_features(raw,kind).loc[lambda f:(f.index>=EVAL_START)&(f.index<EVAL_END)]
        metrics, tr = rollout(agent,fr,return_trace=True)
        np.testing.assert_allclose(metrics['portfolio_return'],experiment['tables']['policy_comparison'].loc[name,'portfolio_return'],rtol=0,atol=1e-12)
        pd.testing.assert_frame_equal(tr,experiment['traces'][name])
        out['recomputed'][name]=metrics
    fr=add_features(raw,experiment['policy'].config.features).loc[lambda f:(f.index>=EVAL_START)&(f.index<EVAL_END)]
    out['evaluation_index']={'first':str(fr.index[0]),'last':str(fr.index[-1]),'rows':len(fr),'continuous':bool((fr.index[1:]-fr.index[:-1]==pd.Timedelta(hours=1)).all()),'first_close':float(fr.close.iloc[0]),'last_close':float(fr.close.iloc[-1])}
    env=make_env(fr,POSITIONS); obs,_=env.reset(seed=123); votes=[]
    try:
        finished=False
        while not finished:
            v=np.array([m.probabilities(obs).argmax() for m in experiment['policy'].agents])
            action=int(v[0]) if np.all(v==v[0]) else POSITIONS.index(0.)
            nxt,reward,term,trunc,info=env.step(action)
            votes.append({'unanimous':bool(np.all(v==v[0])),'action':action,'position':POSITIONS[action], 'votes':v.tolist()})
            obs=nxt;finished=term or trunc
    finally:env.close()
    out['selected_votes']=votes
    out['source_notes']={name:inspect.getsource(obj) for name,obj in [('PPOEnsemble.choose_action_eval',PPOEnsemble.choose_action_eval),('fit',fit),('build_report',build_report),('confirm',confirm)]}
    from gym_trading_env.environments import TradingEnv
    from gym_trading_env.utils.portfolio import Portfolio
    out['simulator_source']={name:inspect.getsource(obj) for name,obj in [('step',TradingEnv.step),('_take_action',TradingEnv._take_action),('reset',TradingEnv.reset),('update_interest',Portfolio.update_interest),('trade_to_position',Portfolio.trade_to_position),('get_portfolio_position',Portfolio.real_position)]}
    import hashlib
    out['same_member_weights']=all(all(torch.equal(a.network.state_dict()[k],b.network.state_dict()[k]) for k in a.network.state_dict()) for a,b in zip(experiment['policy'].agents,experiment['baseline'].agents))
    Path('/Users/alivakof/Desktop/RL/PPO_Presentation/data/verified_experiment.json').write_text(json.dumps(out,indent=2,default=str))
    print('Captured verified experiment; all four evaluation traces exactly match the report.')
    print(experiment['tables']['policy_comparison'].to_string())
    print('Vote counts: unanimous=',sum(x['unanimous'] for x in votes),'disagreement=',sum(not x['unanimous'] for x in votes))
_capture_ppo_presentation()
del _capture_ppo_presentation
'''
c=BlockingKernelClient(connection_file='/Users/alivakof/Library/Jupyter/runtime/kernel-v35601d012827223720e1c084804b8923e4c501e8e.json');c.load_connection_file();c.start_channels()
try:
 c.wait_for_ready(timeout=10);mid=c.execute(code,store_history=False,stop_on_error=True);deadline=time.monotonic()+90
 while time.monotonic()<deadline:
  try:m=c.get_iopub_msg(timeout=5)
  except Exception:continue
  if m.get('parent_header',{}).get('msg_id')!=mid:continue
  if m['msg_type']=='stream':print(m['content']['text'],end='',flush=True)
  if m['msg_type']=='error':raise RuntimeError('\n'.join(m['content']['traceback']))
  if m['msg_type']=='status' and m['content']['execution_state']=='idle':break
 else:raise TimeoutError('Snapshot timed out')
finally:c.stop_channels()
