"""Reproducible DRL tuning using chronological, pre-evaluation validation only.

The assignment evaluation period has already been inspected. Its later comparison
is explicitly a reused holdout, never an input to hyperparameter selection.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import gymnasium as gym
import gym_trading_env  # Registers TradingEnv.

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data/binance-BTCUSDT-1h-20250810-20260917.pkl'
OUT = ROOT / 'eval/tuning'
TRAIN_START = pd.Timestamp('2025-08-17', tz='UTC')
EVAL_START = pd.Timestamp('2026-08-17', tz='UTC')
EVAL_END = pd.Timestamp('2026-09-17', tz='UTC')
# Four equally spaced windows ending 180, 120, 60, and 0 days before evaluation.
# Dates are a schedule rule, not chosen according to candidate performance.
FOLDS = [(EVAL_START-pd.Timedelta(days=d+30), EVAL_START-pd.Timedelta(days=d))
         for d in [180, 120, 60, 0]]
SEEDS = [23, 42, 71]

@dataclass(frozen=True)
class Config:
    name: str
    features: str = 'momentum'
    normalize: bool = True
    reward_scale: float = 100.0
    gamma: float = 0.95
    lr: float = 0.0001
    double: bool = True
    tau: float = 0.01
    turnover_penalty: float = 0.0
    positions: tuple = (-1.0, 0.0, 0.5, 1.0, 2.0)
    hidden: int = 64
    batch: int = 64
    train_every: int = 4
    weight_decay: float = 1e-5

CONFIGS = [
    Config('reference_dqn', features='basic', normalize=False, reward_scale=1,
           gamma=.99, double=False, tau=0, weight_decay=0),
    Config('scaled_reward_ddqn', features='basic'),
    Config('momentum_ddqn'),
    Config('momentum_slow', gamma=.99, lr=3e-5),
    Config('momentum_turnover', turnover_penalty=.0003),
    Config('momentum_unlevered', positions=(-1., 0., 1.), turnover_penalty=.0001),
    Config('momentum_long_only', positions=(0., .5, 1.), turnover_penalty=.0001),
]


def load_data(path=DATA_PATH):
    df = pd.read_pickle(path).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    columns = ['open', 'high', 'low', 'close', 'volume']
    df = df[columns].astype(float)
    expected = pd.date_range(TRAIN_START-pd.Timedelta(days=7), EVAL_END, freq='h', inclusive='left')
    if not df.index.equals(expected) or not np.isfinite(df.to_numpy()).all():
        raise ValueError('Data must contain exactly the expected complete hourly history')
    if (df[['open','high','low','close']] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError('Invalid OHLCV values')
    return df


def add_features(raw, kind='momentum'):
    """Every value uses only the candle at t and earlier candles."""
    df = raw[['open','high','low','close','volume']].copy()
    if kind == 'basic':
        df['feature_close'] = df.close.pct_change()
        for c in ['open','high','low']:
            df['feature_'+c] = df[c]/df.close
        df['feature_volume'] = df.volume/df.volume.rolling(168).max().clip(lower=1e-8)
    elif kind == 'momentum':
        log_close = np.log(df.close)
        returns = log_close.diff()
        for lag in [1, 6, 24, 72, 168]:
            df[f'feature_return_{lag}'] = log_close.diff(lag)
        for window in [24, 168]:
            df[f'feature_volatility_{window}'] = returns.rolling(window).std()
        for window in [24, 72, 168]:
            df[f'feature_trend_{window}'] = df.close/df.close.ewm(span=window, adjust=False, min_periods=window).mean()-1
        df['feature_range'] = (df.high-df.low)/df.close
        df['feature_body'] = df.close/df.open-1
        df['feature_volume'] = np.log((df.volume+1e-8)/(df.volume.rolling(168).mean()+1e-8))
    else:
        raise ValueError(kind)
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def make_env(frame, positions, duration='max'):
    frame = frame.copy()
    frame.index = frame.index.tz_convert('UTC').tz_localize(None)
    return gym.make('TradingEnv', name='BTCUSDT', df=frame, positions=list(positions),
                    initial_position=0, trading_fees=.0001, borrow_interest_rate=.000003,
                    max_episode_duration=duration, verbose=0)


class Replay:
    """Fixed-size array ring buffer avoids repeatedly copying a growing deque."""
    def __init__(self, size, dimensions, rng):
        self.states = np.empty((size, dimensions), np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(size, np.int64)
        self.rewards = np.empty(size, np.float32)
        self.terminals = np.empty(size, np.float32)
        self.size, self.count, self.cursor, self.rng = size, 0, 0, rng
    def append(self, state, action, reward, next_state, terminated):
        i = self.cursor
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.terminals[i] = next_state, terminated
        self.count = min(self.count+1, self.size)
        self.cursor = (i+1) % self.size
    def sample(self, batch):
        i = self.rng.choice(self.count, batch, replace=False)
        return tuple(torch.from_numpy(a[i]) for a in
                     [self.states, self.actions, self.rewards, self.next_states, self.terminals])


class TunedDQN:
    def __init__(self, config, columns, mean, std, seed=23):
        self.config = config
        self.columns = list(columns)
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.positions = tuple(config.positions)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.network = nn.Sequential(nn.Linear(len(mean), config.hidden), nn.ReLU(),
                                     nn.Linear(config.hidden, config.hidden), nn.ReLU(),
                                     nn.Linear(config.hidden, len(config.positions)))
        self.target = deepcopy(self.network).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.lr,
                                          weight_decay=config.weight_decay)
        self.replay = Replay(50000, len(mean), self.rng)
        self.updates = 0

    def encode(self, state):
        values = (np.asarray(state, np.float32)-self.mean)/self.std
        return np.clip(values, -8, 8) if self.config.normalize else values

    def q_values(self, state):
        with torch.no_grad():
            return self.network(torch.from_numpy(self.encode(state))).numpy()

    def choose_action_eval(self, state):
        return int(np.argmax(self.q_values(state)))

    def learn(self):
        cfg = self.config
        states, actions, rewards, next_states, terminals = self.replay.sample(cfg.batch)
        current = self.network(states).gather(1, actions[:,None]).squeeze(1)
        with torch.no_grad():
            future_values = self.target(next_states)
            if cfg.double:
                next_actions = self.network(next_states).argmax(1, keepdim=True)
                future = future_values.gather(1, next_actions).squeeze(1)
            else:
                future = future_values.max(1).values
            target = rewards + cfg.gamma * (1-terminals) * future
        loss = nn.functional.smooth_l1_loss(current, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optimizer.step()
        self.updates += 1
        with torch.no_grad():
            if cfg.tau:
                for target_param, online_param in zip(self.target.parameters(), self.network.parameters()):
                    target_param.lerp_(online_param, cfg.tau)
            elif self.updates % 250 == 0:
                self.target.load_state_dict(self.network.state_dict())
        return float(loss.item())

    def payload(self):
        return dict(config=asdict(self.config), columns=self.columns, mean=self.mean.tolist(),
                    std=self.std.tolist(), seed=self.seed, weights=self.network.state_dict())

    @classmethod
    def from_payload(cls, data):
        cfg = Config(**{**data['config'], 'positions': tuple(data['config']['positions'])})
        agent = cls(cfg, data['columns'], data['mean'], data['std'], data['seed'])
        agent.network.load_state_dict(data['weights'])
        agent.target.load_state_dict(data['weights'])
        agent.network.eval()
        return agent


class EnsemblePolicy:
    def __init__(self, agents):
        if not agents:
            raise ValueError('At least one agent is required')
        self.agents = agents
        self.positions = agents[0].positions
        self.config = agents[0].config
        self.columns = agents[0].columns
        if any(a.positions != self.positions or a.config != self.config or a.columns != self.columns for a in agents):
            raise ValueError('Ensemble members must share configuration and feature order')
    def choose_action_eval(self, state):
        return int(np.argmax(np.mean([a.q_values(state) for a in self.agents], axis=0)))
    def save(self, path):
        torch.save({'format':1, 'agents':[a.payload() for a in self.agents]}, path)
    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        return cls([TunedDQN.from_payload(a) for a in data['agents']])


class ValidatedEnsemblePolicy(EnsemblePolicy):
    """Validation-selected agreement rule; training/checkpoints remain unchanged."""
    METHODS = ('mean_q', 'majority', 'unanimous', 'normalized_advantage')

    def __init__(self, agents, aggregation='mean_q'):
        super().__init__(agents)
        if aggregation not in self.METHODS:
            raise ValueError(aggregation)
        self.aggregation = aggregation

    def choose_action_eval(self, state):
        q = np.asarray([a.q_values(state) for a in self.agents])
        if self.aggregation == 'mean_q':
            return int(q.mean(axis=0).argmax())
        if self.aggregation == 'normalized_advantage':
            q = (q-q.mean(axis=1, keepdims=True))/np.maximum(q.std(axis=1, keepdims=True), 1e-6)
            return int(q.mean(axis=0).argmax())
        votes = q.argmax(axis=1)
        counts = np.bincount(votes, minlength=len(self.positions))
        action = int(counts.argmax())
        required = len(self.agents) if self.aggregation == 'unanimous' else len(self.agents)//2+1
        return action if counts[action] >= required else self.positions.index(0.)

    def save(self, path):
        torch.save(dict(format=2, aggregation=self.aggregation,
                        agents=[a.payload() for a in self.agents]), path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        return cls([TunedDQN.from_payload(a) for a in data['agents']], data.get('aggregation','mean_q'))


def fit(frame, cfg, seed, steps=24000):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    columns = [c for c in frame if c.startswith('feature_')]
    if cfg.normalize:
        mean = np.r_[frame[columns].mean().values, [0,0]]
        std = np.r_[frame[columns].std().clip(lower=1e-6).values, [1,1]]
    else:
        mean, std = np.zeros(len(columns)+2), np.ones(len(columns)+2)
    agent = TunedDQN(cfg, columns, mean, std, seed)
    env = make_env(frame, cfg.positions, duration=min(720, len(frame)-1))
    total_steps, episode = 0, 0
    logs = []
    try:
        while total_steps < steps:
            # Installed TradingEnv uses global numpy RNG for random start locations.
            np.random.seed(seed+episode)
            state, _ = env.reset(seed=seed+episode)
            terminated = truncated = False
            raw_total, losses = 0.0, []
            while not (terminated or truncated) and total_steps < steps:
                epsilon = max(.05, 1-total_steps/(.60*steps))
                action = int(agent.rng.integers(len(cfg.positions))) if agent.rng.random() < epsilon else agent.choose_action_eval(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                # TradingEnv's insolvency transition returns zero; explicitly penalize it for learning.
                learning_reward = (-1.0 if terminated else float(reward))
                learning_reward -= cfg.turnover_penalty * abs(cfg.positions[action]-float(state[-2]))
                learning_reward *= cfg.reward_scale
                if not np.isfinite(learning_reward):
                    raise FloatingPointError('Non-finite reward')
                agent.replay.append(agent.encode(state), action, learning_reward,
                                    (np.zeros_like(agent.mean) if terminated else agent.encode(next_state)), terminated)
                total_steps += 1
                if agent.replay.count >= 256 and total_steps % cfg.train_every == 0:
                    losses.append(agent.learn())
                state = next_state
                raw_total += reward
            logs.append(dict(episode=episode+1, steps=total_steps, raw_log_return=raw_total,
                             mean_loss=float(np.mean(losses)) if losses else 0, epsilon=epsilon,
                             updates=agent.updates))
            episode += 1
    finally:
        env.close()
    return agent, pd.DataFrame(logs)


def rollout(policy, frame, positions=None, return_trace=False, render_dir=None):
    positions = policy.positions if positions is None else positions
    env = make_env(frame, positions)
    state, _ = env.reset(seed=123)
    records = [dict(timestamp=frame.index[0], portfolio_value=1000., position=0., reward=0.)]
    try:
        done = False
        while not done:
            action = policy.choose_action_eval(state)
            state, reward, terminated, truncated, info = env.step(action)
            records.append(dict(timestamp=frame.index[len(records)],
                                portfolio_value=float(info['portfolio_valuation']),
                                position=float(positions[action]), reward=float(reward)))
            done = terminated or truncated
        if render_dir:
            Path(render_dir).mkdir(parents=True, exist_ok=True)
            env.unwrapped.save_for_render(dir=str(render_dir))
    finally:
        env.close()
    trace = pd.DataFrame(records)
    values = trace.portfolio_value.to_numpy()
    market_return = float(frame.close.iloc[-1]/frame.close.iloc[0]-1)
    ret = values[-1]/values[0]-1
    dd = float(np.min(values/np.maximum.accumulate(values)-1))
    metrics = dict(portfolio_return=ret, market_return=market_return,
                   excess_return=ret-market_return, max_drawdown=dd,
                   log_return=float(np.log(max(values[-1]/values[0],1e-12))),
                   action_changes=int(np.count_nonzero(np.diff(trace.position))),
                   steps=len(trace)-1, total_reward=float(trace.reward.sum()))
    return (metrics, trace) if return_trace else metrics


class ConstantPolicy:
    def __init__(self, positions, position):
        self.positions = positions
        self.action = positions.index(position)
    def choose_action_eval(self, state):
        return self.action


def job(cfg_dict, seed, fold_index, steps, phase):
    cfg = Config(**{**cfg_dict, 'positions':tuple(cfg_dict['positions'])})
    frame = add_features(load_data().loc[lambda x: x.index < EVAL_START], cfg.features)
    start, end = FOLDS[fold_index]
    training = frame.loc[(frame.index >= TRAIN_START) & (frame.index < start)]
    validation = frame.loc[(frame.index >= start) & (frame.index < end)]
    assert training.index.max() < validation.index.min() and validation.index.max() < EVAL_START
    agent, history = fit(training, cfg, seed, steps)
    metrics = rollout(agent, validation)
    path = OUT / phase / f'{cfg.name}_fold{fold_index}_seed{seed}'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.payload(), path.with_suffix('.pth'))
    history.to_csv(path.with_suffix('.training.csv'), index=False)
    row = dict(config=cfg.name, seed=seed, fold=fold_index, start=str(start), end=str(end),
               training_steps=steps, updates=agent.updates, **metrics)
    path.with_suffix('.json').write_text(json.dumps(row, indent=2))
    return row


def run_jobs(configs, seeds, steps, phase, workers=3):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for cfg in configs:
            for fold in range(len(FOLDS)):
                for seed in seeds:
                    path = OUT / phase / f'{cfg.name}_fold{fold}_seed{seed}.json'
                    if path.exists():
                        row = json.loads(path.read_text())
                        if row['training_steps'] != steps:
                            raise ValueError('Existing trial has a different budget; use a fresh output directory')
                        rows.append(row)
                    else:
                        futures.append(pool.submit(job, asdict(cfg), seed, fold, steps, phase))
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(phase, row['config'], 'fold',row['fold'],'seed',row['seed'],
                  'return',f"{row['portfolio_return']:.2%}",'drawdown',f"{row['max_drawdown']:.2%}",flush=True)
            pd.DataFrame(rows).to_csv(OUT/f'{phase}_trials.csv', index=False)
    result = pd.DataFrame(rows).sort_values(['config','fold','seed'])
    result.to_csv(OUT/f'{phase}_trials.csv', index=False)
    return result


def ranking(results):
    # Positive drawdown penalty and seed/fold variability penalty. All weights fixed before training.
    summary = results.groupby('config').agg(mean_log_return=('log_return','mean'),
               std_log_return=('log_return','std'), mean_return=('portfolio_return','mean'),
               worst_return=('portfolio_return','min'), mean_drawdown=('max_drawdown','mean'),
               mean_action_changes=('action_changes','mean'))
    summary['selection_score'] = summary.mean_log_return + .25*summary.mean_drawdown - .10*summary.std_log_return
    return summary.sort_values('selection_score', ascending=False)


def implementation_fingerprint():
    parts = [Config, load_data, add_features, make_env, Replay, TunedDQN,
             EnsemblePolicy, fit, rollout, ranking]
    return hashlib.sha256('\n'.join(inspect.getsource(p) for p in parts).encode()).hexdigest()


def validate_trials(results, configs, seeds, steps):
    expected = {(c.name, f, s) for c in configs for f in range(len(FOLDS)) for s in seeds}
    keys = list(zip(results.config, results.fold, results.seed))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Trial grid is incomplete or contains unexpected/duplicate trials')
    if not results.training_steps.eq(steps).all():
        raise ValueError('Trial budgets differ from the registered protocol')
    if not np.isfinite(results[['portfolio_return','max_drawdown','log_return']].to_numpy()).all():
        raise ValueError('Non-finite validation metrics')


def protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_data().loc[lambda x: x.index < EVAL_START]
    digest = hashlib.sha256(pd.util.hash_pandas_object(raw, index=True).values.tobytes()).hexdigest()
    description = dict(configurations=[asdict(c) for c in CONFIGS],
                       folds=[[str(a),str(b)] for a,b in FOLDS], screening_seeds=[23],
                       confirmation_seeds=SEEDS, screening_steps=12000, confirmation_steps=24000,
                       final_steps=24000, pre_eval_data_sha256=digest,
                       implementation_sha256=implementation_fingerprint(),
                       score='mean(log return) - 0.25*mean(drawdown magnitude) - 0.10*std(log return)',
                       final_policy='mean Q-values of three seeds, selected by validation ensemble score',
                       reused_evaluation=True)
    path = OUT/'protocol.json'
    encoded = json.dumps(description, indent=2)
    if path.exists() and json.loads(path.read_text()) != json.loads(encoded):
        raise ValueError('Protocol changed. Preserve old results and start a new output directory.')
    path.write_text(encoded)
    return description


def confirm(workers):
    screening = pd.read_csv(OUT/'screening_trials.csv')
    validate_trials(screening, CONFIGS, [23], 12000)
    ranks = ranking(screening)
    ranks.to_csv(OUT/'screening_summary.csv')
    finalists = ranks.index[:2].tolist()
    configs = [c for c in CONFIGS if c.name in finalists]
    results = run_jobs(configs, SEEDS, 24000, 'confirmation', workers)
    validate_trials(results, configs, SEEDS, 24000)
    ranking(results).to_csv(OUT/'confirmation_individual_summary.csv')
    ensemble_rows = []
    raw = load_data().loc[lambda x: x.index < EVAL_START]
    for cfg in configs:
        frame = add_features(raw, cfg.features)
        for fold,(start,end) in enumerate(FOLDS):
            members = [TunedDQN.from_payload(torch.load(OUT/'confirmation'/f'{cfg.name}_fold{fold}_seed{seed}.pth',
                         map_location='cpu', weights_only=True)) for seed in SEEDS]
            ensemble = EnsemblePolicy(members)
            metrics = rollout(ensemble, frame.loc[(frame.index>=start)&(frame.index<end)])
            ensemble_rows.append(dict(config=cfg.name, fold=fold, **metrics))
    ensembles = pd.DataFrame(ensemble_rows)
    ensembles.to_csv(OUT/'confirmation_ensembles.csv',index=False)
    ranked = ranking(ensembles)
    ranked.to_csv(OUT/'confirmation_ensemble_summary.csv')
    choice = ranked.index[0]
    selected = next(c for c in configs if c.name==choice)
    (OUT/'selected_configuration.json').write_text(json.dumps(dict(config=asdict(selected), seeds=SEEDS,
          training_steps=24000, selection='best validation ensemble score across four chronological folds',
          evaluation_used_for_selection=False, reused_assignment_holdout=True),indent=2))
    print('FROZEN SELECTION',choice,flush=True)
    return selected


def validate_aggregation_protocol():
    amendment = json.loads((OUT/'aggregation_protocol.json').read_text())
    current = hashlib.sha256(inspect.getsource(ValidatedEnsemblePolicy).encode()).hexdigest()
    if amendment.get('implementation_sha256') != current:
        raise ValueError('Agreement-rule implementation changed; register a new amendment before rerunning')
    return amendment


def aggregate_confirmations():
    """Small registered follow-up to observed ensemble instability, before test review."""
    amendment = validate_aggregation_protocol()
    configs = [c for c in CONFIGS if c.name in amendment['configurations']]
    raw = load_data().loc[lambda x:x.index<EVAL_START]
    rows = []
    for cfg in configs:
        frame = add_features(raw, cfg.features)
        for fold,(start,end) in enumerate(FOLDS):
            members = [TunedDQN.from_payload(torch.load(OUT/'confirmation'/f'{cfg.name}_fold{fold}_seed{seed}.pth',
                        map_location='cpu',weights_only=True)) for seed in SEEDS]
            part = frame.loc[(frame.index>=start)&(frame.index<end)]
            for method in amendment['aggregations']:
                policy = ValidatedEnsemblePolicy(members, method)
                rows.append(dict(config=cfg.name, aggregation=method, fold=fold, **rollout(policy,part)))
    trials = pd.DataFrame(rows)
    trials.to_csv(OUT/'aggregation_validation.csv',index=False)
    summary = trials.groupby(['config','aggregation']).agg(
        mean_log_return=('log_return','mean'), std_log_return=('log_return','std'),
        mean_return=('portfolio_return','mean'), worst_return=('portfolio_return','min'),
        mean_drawdown=('max_drawdown','mean'), mean_action_changes=('action_changes','mean'))
    summary['selection_score'] = summary.mean_log_return + .25*summary.mean_drawdown - .10*summary.std_log_return
    summary = summary.sort_values('selection_score',ascending=False)
    summary.to_csv(OUT/'aggregation_summary.csv')
    config_name,method = summary.index[0]
    cfg = next(c for c in configs if c.name==config_name)
    selected_path = OUT/'selected_configuration.json'
    # Preserve the original mean-Q decision and document the validation-only amendment.
    initial = OUT/'selected_configuration_mean_q.json'
    if not initial.exists(): initial.write_text(selected_path.read_text())
    selected_path.write_text(json.dumps(dict(config=asdict(cfg), aggregation=method,
        seeds=SEEDS, training_steps=24000,
        selection='best validation score across four folds after registered agreement-rule comparison',
        evaluation_used_for_selection=False, reused_assignment_holdout=True),indent=2))
    print('FROZEN AGGREGATION SELECTION',config_name,method,flush=True)
    return cfg,method


def final_fit_job(config_dict, seed, steps):
    cfg=Config(**{**config_dict,'positions':tuple(config_dict['positions'])})
    directory=OUT/'final_members'
    directory.mkdir(exist_ok=True)
    path=directory/f'{cfg.name}_seed{seed}.pth'
    history_path=directory/f'{cfg.name}_seed{seed}_training.csv'
    # Reuse the same final fits when only the aggregation rule changes.
    candidates=[(path,history_path),(OUT/f'final_seed{seed}.pth',OUT/f'final_seed{seed}_training.csv')]
    for weights,history_file in candidates:
        if weights.exists() and history_file.exists():
            data=torch.load(weights,map_location='cpu',weights_only=True)
            history=pd.read_csv(history_file)
            loaded_cfg=Config(**{**data['config'],'positions':tuple(data['config']['positions'])})
            if loaded_cfg==cfg and data['seed']==seed and int(history.steps.iloc[-1])==steps:
                if weights!=path:
                    torch.save(data,path)
                    history.to_csv(history_path,index=False)
                return str(path)
    frame=add_features(load_data().loc[lambda x:x.index<EVAL_START],cfg.features)
    training=frame.loc[frame.index>=TRAIN_START]
    agent,history=fit(training,cfg,seed,steps)
    torch.save(agent.payload(),path)
    history.to_csv(history_path,index=False)
    return str(path)


def finalize(workers):
    selected=json.loads((OUT/'selected_configuration.json').read_text())
    if 'aggregation' in selected: validate_aggregation_protocol()
    cfg=Config(**{**selected['config'],'positions':tuple(selected['config']['positions'])})
    # Train only before EVAL_START. The final evaluation occurs after all choices are frozen.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        paths=list(pool.map(final_fit_job,[asdict(cfg)]*3,SEEDS,[selected['training_steps']]*3))
    policy=ValidatedEnsemblePolicy([TunedDQN.from_payload(torch.load(p,map_location='cpu',weights_only=True)) for p in paths],
                                   selected.get('aggregation','mean_q'))
    policy.save(OUT/'tuned_trading_policy.pth')
    raw=load_data()
    features=add_features(raw,cfg.features)
    evaluation=features.loc[(features.index>=EVAL_START)&(features.index<EVAL_END)]
    assert len(evaluation)==744
    metrics,trace=rollout(policy,evaluation,return_trace=True,render_dir=OUT/'render_logs')
    trace.to_csv(OUT/'evaluation_trace.csv',index=False)
    rows=[]
    for episode in range(10):
        m=metrics if episode==0 else rollout(policy,evaluation)
        rows.append(dict(episode=episode+1, **{k:m[k] for k in ['portfolio_return','market_return','excess_return','steps','total_reward']}))
    pd.DataFrame(rows).to_csv(OUT/'evaluation_results.csv',index=False)
    (OUT/'final_metrics.json').write_text(json.dumps(metrics,indent=2))
    benchmark=[]
    for position in [0.,1.]:
        m=rollout(ConstantPolicy((0.,1.),position),evaluation)
        benchmark.append(dict(policy='cash' if position==0 else 'buy_and_hold',**m))
    pd.DataFrame(benchmark).to_csv(OUT/'evaluation_benchmarks.csv',index=False)
    print('FINAL REUSED HOLDOUT',metrics,flush=True)


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['screen','confirm','aggregate','final'])
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    protocol()
    if args.phase=='screen':
        result=run_jobs(CONFIGS,[23],12000,'screening',args.workers)
        ranking(result).to_csv(OUT/'screening_summary.csv')
        print(ranking(result))
    elif args.phase=='confirm':confirm(args.workers)
    elif args.phase=='aggregate':aggregate_confirmations()
    else:finalize(args.workers)

if __name__=='__main__':main()
