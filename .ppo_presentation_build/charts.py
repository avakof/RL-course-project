from pathlib import Path
import json,ast
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter, FuncFormatter
R=Path('/Users/alivakof/Desktop/RL/PPO_Presentation');J=json.loads((R/'data/verified_experiment.json').read_text())
def frame(v):return pd.DataFrame(v['data'],index=v['index'],columns=v['columns'])
T={k:frame(v) for k,v in J['tables'].items()};TR={k:frame(v) for k,v in J['traces'].items()}
for v in TR.values():v['timestamp']=pd.to_datetime(v.timestamp,utc=True)
C={'Baseline PPO':'#2166AC','Selected PPO':'#C66118','Cash':'#777C85','Buy and hold':'#237A57'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':15,'axes.titleweight':'bold','axes.labelcolor':'#243746','text.color':'#142735','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#C3CDD5','grid.color':'#DFE5EA','axes.axisbelow':True,'svg.fonttype':'none','savefig.facecolor':'white'})
CAP={}
def save(fig,name,caption):
 fig.savefig(R/'charts'/f'{name}.png',dpi=210,bbox_inches='tight');fig.savefig(R/'charts'/f'{name}.svg',bbox_inches='tight');plt.close(fig);CAP[name]=caption
# Market history and exact folds.
d=frame(J['daily_close']);d.index=pd.to_datetime(d.index,utc=True)
f,ax=plt.subplots(2,1,figsize=(12,6.7),gridspec_kw={'height_ratios':[2,1.4]},layout='constrained')
ax[0].plot(d.index,d.close,color='#2166AC',lw=1);ax[0].set(yscale='log',ylabel='BTC/USDT close (log scale)',title='Market history spans 2020–2026; validation covers four recent windows')
ax[0].yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:,.0f}'));ax[0].grid(axis='y')
for i,(a,b) in enumerate(J['protocol']['folds']):
 a,b=pd.Timestamp(a),pd.Timestamp(b);ax[0].axvspan(a,b,color='#C66118',alpha=.15)
 ax[1].barh(3-i,(b-a).days,left=mdates.date2num(a),color='#C66118',height=.52)
 ax[1].text(mdates.date2num(a)+1,3-i,f'{a:%d %b}–{b:%d %b}',va='center',fontsize=10,color='white')
a=pd.Timestamp(J['protocol']['evaluation_start']);b=pd.Timestamp(J['protocol']['evaluation_end']);ax[0].axvspan(a,b,color='#237A57',alpha=.3)
ax[1].barh(-1,(b-a).days,left=mdates.date2num(a),color='#237A57',height=.52);ax[1].text(mdates.date2num(a)+1,-1,'17 Aug–17 Sep',va='center',fontsize=9,color='white')
ax[1].set_yticks([3,2,1,0,-1],['Validation 1','Validation 2','Validation 3','Validation 4','Evaluation']);ax[1].set_xlim(mdates.date2num(pd.Timestamp('2026-01-01')),mdates.date2num(pd.Timestamp('2026-10-01')));ax[1].xaxis_date();ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'));ax[1].grid(axis='x');ax[1].set_title('Expanding training history ends before each validation window',fontsize=12)
save(f,'01_data_and_windows','Daily closing prices are plotted from the verified hourly dataset. Shaded windows show four 30-day validations and the 31-day evaluation. Date endpoints are exclusive; all timestamps are UTC.')
# Action space.
f,ax=plt.subplots(figsize=(11,4.2),layout='constrained');pos=np.array(J['protocol']['configurations'][0]['positions']);ax.bar(np.arange(9),pos,color=['#AE4E56' if p<0 else '#777C85' if p==0 else '#237A57' for p in pos],width=.6);ax.axhline(0,color='#354F61',lw=1);ax.set_xticks(range(9),[f'{p:g}' for p in pos]);ax.set(ylabel='Requested target exposure',xlabel='Nine discrete positions (action indices 0–8; cash index 4)',title='A categorical policy chooses fractional short, cash or long exposure');ax.set_yticks(pos);ax.grid(axis='y')
save(f,'02_action_space','Position values are target exposures, not buy/sell quantities or a continuous action space. Repeating a target does not force rebalancing in the installed simulator; realised exposure can drift.')
# Screening four fold outcomes and mean.
s=T['screening_summary'].sort_values('mean_return');raw=T['screening_trials'];f,ax=plt.subplots(figsize=(11,5.8),layout='constrained')
labels=[x.replace('ppo_','').replace('_',' ') for x in s.index];y=np.arange(len(s));ax.barh(y,100*s.mean_return,color=['#2166AC' if x=='ppo_baseline' else '#9EBACB' for x in s.index],height=.6,label='Mean of four folds')
for i,k in enumerate(s.index):ax.scatter(raw.loc[raw.config.eq(k),'portfolio_return']*100,[i]*4,s=24,color='#1D3444',zorder=3)
ax.axvline(0,color='#667D8C',lw=1);ax.set_yticks(y,labels);ax.set(xlabel='Validation portfolio return (%)',title='Initial screening: momentum led, but outcomes varied by fold');ax.grid(axis='x')
for i,v in enumerate(s.mean_return*100):ax.text(7.8,i,f'{v:+.2f}%',va='center',ha='right',fontsize=10)
ax.set_xlim(min(-32,100*raw.portfolio_return.min()-2),9)
save(f,'03_initial_screening','Each point is one validation fold. Bars are arithmetic means of four fold returns, seed 23, 16,384 training transitions per fit. These are validation results, not final evaluation returns.')
# confirmation comparison.
keys=['ppo_baseline','ppo_momentum','ppo_low_turnover'];ss=T['screening_summary'].loc[keys];cc=T['confirmation_individual_summary'].loc[keys];f,ax=plt.subplots(figsize=(10.6,4.8),layout='constrained');x=np.arange(3)
ax.bar(x-.18,100*ss.mean_return,width=.35,color='#9EBACB',label='Screening: 1 seed × 4 folds');ax.bar(x+.18,100*cc.mean_return,width=.35,color='#2166AC',label='Confirmation: 3 seeds × 4 folds');ax.axhline(0,color='#667D8C',lw=1);ax.set_xticks(x,['Baseline','Momentum','Low turnover']);ax.set(ylabel='Mean validation return (%)',title='The initial gains did not persist across the confirmation stage');ax.grid(axis='y');ax.legend(loc='lower right',fontsize=10)
for delta,ser in [(-.18,ss.mean_return),(.18,cc.mean_return)]:
 for i,v in enumerate(100*ser):ax.text(i+delta,v+(.25 if v>=0 else -.3),f'{v:+.2f}%',ha='center',va='bottom' if v>=0 else 'top',fontsize=10)
ax.set_ylim(-13,3)
save(f,'04_confirmation','Confirmation refits each candidate from scratch using 49,152 transitions and seeds 23, 42 and 71. Both seed coverage and training budget changed, so the shift is not a controlled estimate of either effect alone.')
# ranking.
s=T['ensemble_summary'].copy();s.loc['Cash']=0;s=s.sort_values('selection_score');f,ax=plt.subplots(figsize=(11.7,5.4),layout='constrained')
labels=[x.replace('ppo_','').replace('::mean_probability',' / mean probability').replace('::unanimous',' / unanimous').replace('_',' ') for x in s.index]
colors=['#777C85' if x=='Cash' else '#C66118' if x=='ppo_baseline::unanimous' else '#2166AC' for x in s.index]
ax.barh(np.arange(len(s)),s.selection_score,color=colors,height=.6);ax.set_yticks(range(len(s)),labels);ax.axvline(0,color='#667D8C',lw=1);ax.grid(axis='x');ax.set(xlabel='Validation selection score (higher is better; not return %)',title='Unanimity won the PPO search; cash had a higher score')
for i,v in enumerate(s.selection_score):ax.text(.002,i,f'{v:.6f}',va='center',fontsize=10)
ax.set_xlim(-.12,.025)
save(f,'05_validation_selection','Score = mean(log return) − 0.25 × mean(drawdown magnitude) − 0.10 × sample standard deviation(log return). Scores use four ensemble backtests, one per fold. Cash is shown for context but was excluded from selectable PPO candidates.')
# folds.
v=T['validation_fold_comparison'];f,ax=plt.subplots(figsize=(11.7,5.2),layout='constrained');x=np.arange(4);series=[('Buy and hold','Buy and hold'),('ppo_baseline::mean_probability','Baseline PPO'),('ppo_baseline::unanimous','Selected PPO')]
for offset,(key,label) in zip([-.25,0,.25],series):ax.bar(x+offset,100*v[key],width=.24,label=label,color=C[label])
ax.axhline(0,color=C['Cash'],lw=1.5,linestyle='--',label='Cash');ax.set_xticks(x,['19 Jan–18 Feb','20 Mar–19 Apr','19 May–18 Jun','18 Jul–17 Aug']);ax.set(ylabel='Validation portfolio return (%)',title='Three of four validation windows declined for buy-and-hold');ax.grid(axis='y');ax.legend(loc='lower right',fontsize=10)
save(f,'06_validation_folds','All dates are in 2026 UTC, end exclusive. Each fold has 720 observations and 719 transitions. The selected policy’s fold returns were 0.00%, +0.48%, −0.24% and −0.09%.')
# maincomparison.
co=T['policy_comparison'];order=['Baseline PPO','Selected PPO','Cash','Buy and hold'];f,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for axis,vals,title in [(ax[0],100*co.loc[order,'portfolio_return'],'Evaluation net return (%)'),(ax[1],100*co.loc[order,'max_drawdown'].abs(),'Maximum drawdown magnitude (%)')]:
 axis.bar(range(4),vals,color=[C[k] for k in order]);axis.set_xticks(range(4),['Baseline\nPPO','Selected\nPPO','Cash','Buy and\nhold']);axis.axhline(0,color='#667D8C',lw=1);axis.set_title(title);axis.grid(axis='y')
 for i,v in enumerate(vals):axis.annotate(f'{v:+.2f}%' if axis==ax[0] else f'{v:.2f}%',(i,v),xytext=(0,6 if v>=0 else -15),textcoords='offset points',ha='center',fontsize=12,fontweight='bold')
ax[0].set_ylim(-3.5,25.2);ax[1].set_ylim(0,9.3)
save(f,'07_evaluation_comparison','Evaluation candle labels: 17 Aug 2026 00:00 through 16 Sep 2026 23:00 UTC. Returns use first-to-last included close, with simulator fees. Borrowing accounting has a verified limitation; slippage and spread are omitted.')
# paths.
f,ax=plt.subplots(2,1,figsize=(12,7.3),sharex=True,gridspec_kw={'height_ratios':[1.6,1]},layout='constrained')
for k in order:
 t=TR[k];v=t.portfolio_value.to_numpy();ax[0].plot(t.timestamp,100*(v/v[0]-1),label=k,color=C[k],lw=1.8);ax[1].plot(t.timestamp,100*(v/np.maximum.accumulate(v)-1),label=k,color=C[k],lw=1.5)
ax[0].set(title='The baseline participated in the rising evaluation market',ylabel='Cumulative return (%)');ax[1].set(ylabel='Drawdown (%)',xlabel='UTC candle-open timestamp');ax[0].legend(ncol=4,loc='upper left',fontsize=10);ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'));ax[0].grid();ax[1].grid();save(f,'08_equity_and_drawdown','All paths start at 1,000 USDT in cash. Selected PPO ends at 994.29; baseline at 1,144.43. Curves are recomputed from the live trained agents and exactly match the stored experiment traces.')
# allocation.
f,ax=plt.subplots(figsize=(12,5.5),layout='constrained');x=np.arange(9)
for off,k in [(-.19,'Baseline PPO'),(.19,'Selected PPO')]:
 counts=TR[k].position.iloc[1:].value_counts().reindex(pos,fill_value=0);percent=100*counts/743;ax.bar(x+off,percent,.37,label=k,color=C[k])
 for i,v in enumerate(percent):
  if v>0:ax.text(i+off,v+1.3,f'{v:.2f}%',ha='center',fontsize=9,rotation=0)
ax.set_xticks(x,[f'{p:g}' for p in pos]);ax.set(ylim=(0,110),xlabel='Chosen target position',ylabel='Share of 743 decisions (%)',title='Selected PPO chose cash on 742 of 743 decisions');ax.legend(loc='upper left');ax.grid(axis='y');save(f,'09_exposure_distribution','Direct replay confirms all 742 cash decisions were caused by disagreement between the three models. They unanimously chose +1 only once, producing one holding interval and two position changes (entry and exit). These are decision frequencies, not capital weights.')
# member results.
m=T['final_seed_results'].loc[lambda d:d.policy.eq('Baseline PPO')];f,ax=plt.subplots(figsize=(11,4.8),layout='constrained');ax.bar(range(3),100*m.portfolio_return,color=['#AE4E56','#779DAC','#2166AC'],width=.55)
for i,v in enumerate(100*m.portfolio_return):ax.text(i,v+(.65 if v>0 else -.65),f'{v:+.2f}%',ha='center',va='bottom' if v>0 else 'top',fontweight='bold')
ax.axhline(100*co.loc['Baseline PPO','portfolio_return'],color=C['Baseline PPO'],linestyle='--',label='Mean-probability ensemble: +14.44%');ax.axhline(100*co.loc['Selected PPO','portfolio_return'],color=C['Selected PPO'],linestyle=':',label='Unanimous ensemble: −0.57%');ax.set_xticks(range(3),[f'Seed {s}' for s in m.seed]);ax.set(ylim=(-20,24),ylabel='Evaluation portfolio return (%)',title='Individual seeds differ substantially');ax.grid(axis='y');ax.legend(loc='upper left',fontsize=10);save(f,'10_seed_sensitivity','These are the same three fitted networks used by both ensembles. The six notebook rows repeat these three members under two policy labels; they are not six independent models. An ensemble combines decisions, not realised returns.')
# diagnostics.
f,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');metrics=['entropy','value_loss','approx_kl','explained_variance'];labels=['Policy entropy (nats)','Critic half-MSE (scaled reward targets)','Approximate KL (accepted minibatches)','Critic explained variance']
for i,(key,h) in enumerate(J['histories'].items()):
 h=frame(h);seed=ast.literal_eval(key)[2]
 for ax,m,l in zip(axes.flat,metrics,labels):ax.plot(h.steps/1000,h[m],lw=1.3,label=f'Seed {seed}');ax.set(xlabel='Training transitions (thousands)',title=l);ax.grid(alpha=.7)
axes[0,0].axhline(np.log(9),color='#777C85',ls=':',lw=1);axes[1,1].axhline(0,color='#777C85',lw=1);axes[0,0].legend(fontsize=9);save(f,'11_training_diagnostics','Final baseline networks: 48 rollout batches × 1,024 transitions = 49,152 per seed; 1,152 minibatch optimiser updates per seed. No KL early stop occurred. Entropy declines, but noisy critic explained variance does not establish predictive skill or convergence.')
(R/'data/chart_captions.json').write_text(json.dumps(CAP,indent=2))
print('Created',len(CAP),'charts in PNG and SVG.')
