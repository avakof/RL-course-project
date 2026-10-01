"""Discrete-action PPO and reproducible trading experiments for Project_PPO.ipynb.

All hyperparameter selection uses pre-assignment chronological validation.
The assignment month has already been viewed in the DQN project: it is a reused
holdout comparison. Expanded-history artifacts go to eval/ppo_202004_nine_actions/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from torch import nn
from torch.distributions import Categorical

from trading_tuning import (ROOT, FOLDS, SEEDS, add_features as shared_add_features,
                           make_env, rollout, ConstantPolicy)
from btc_data import DATA_PATH, TRAIN_START, EVAL_START, EVAL_END, load_data

OUT = ROOT / 'eval/ppo_202004_nine_actions'
POSITIONS = (-1., -.75, -.5, -.25, 0., .25, .5, .75, 1.)
SCREEN_STEPS = 16384
CONFIRM_STEPS = 49152


@dataclass(frozen=True)
class PPOConfig:
    name: str
    features: str = 'momentum'
    positions: tuple = POSITIONS
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip_range: float = .2
    entropy_coef: float = .01
    value_coef: float = .5
    max_grad_norm: float = .5
    target_kl: float = .02
    epochs: int = 6
    batch_size: int = 256
    n_envs: int = 4
    rollout_steps: int = 256
    reward_scale: float = 100.
    turnover_penalty: float = 0.
    anneal_lr: bool = True


# Frozen before inspecting PPO validation or assignment-period performance.
CONFIGS = [
    PPOConfig('ppo_baseline', features='basic', entropy_coef=0., anneal_lr=False),
    PPOConfig('ppo_momentum'),
    PPOConfig('ppo_cautious', lr=1e-4, clip_range=.1, entropy_coef=.005),
    PPOConfig('ppo_short_horizon', gamma=.95, gae_lambda=.90),
    PPOConfig('ppo_turnover', gamma=.95, turnover_penalty=.0003),
    PPOConfig('ppo_low_turnover', turnover_penalty=.0001),
    PPOConfig('ppo_exploration', entropy_coef=.03, turnover_penalty=.0001),
]
AGGREGATIONS = ('mean_probability', 'unanimous')


def add_features(raw, kind='momentum'):
    """Restart rolling features at missing hours; never impute market candles."""
    segments = raw.index.to_series().diff().ne(pd.Timedelta(hours=1)).cumsum()
    return pd.concat([shared_add_features(part, kind) for _, part in raw.groupby(segments)])


class TrainingWindowEnv(gym.Wrapper):
    """Sample complete hourly windows; neither features nor episodes cross gaps."""
    def __init__(self, frame, positions, duration):
        self.frame, self.positions, self.duration = frame, positions, duration
        times = frame.index.asi8
        # Ordered unique timestamps mean equal endpoint spacing implies no interior gaps.
        if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
            raise ValueError('Training timestamps must be ordered and unique')
        self.starts = np.flatnonzero(times[duration-1:] - times[:len(times)-duration+1]
                                     == (duration-1)*pd.Timedelta(hours=1).value)
        if not len(self.starts):
            raise ValueError('No continuous hourly training window of the requested duration')
        super().__init__(make_env(frame.iloc[:duration], positions))

    def reset(self, *, seed=None, options=None):
        start = int(np.random.choice(self.starts))
        self.env.close()
        self.env = make_env(self.frame.iloc[start:start+self.duration], self.positions)
        return self.env.reset(seed=seed, options=options)


def config_from_dict(data):
    return PPOConfig(**{**data, 'positions': tuple(data['positions'])})


def compute_gae(rewards, values, next_values, terminated, done, gamma, gae_lambda):
    """Time-major [T, N] GAE; bootstrap truncations but never cross resets.

    `next_values` must refer to the transition's final observation, not the
    reset observation. A bankruptcy/true terminal has no future value.
    """
    arrays = [np.asarray(a, np.float32) for a in
              (rewards, values, next_values, terminated, done)]
    rewards, values, next_values, terminated, done = arrays
    if rewards.ndim != 2 or any(a.shape != rewards.shape for a in arrays):
        raise ValueError('GAE inputs must have matching [time, environment] shapes')
    advantages = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[1], np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1-terminated[t]) * next_values[t] - values[t]
        carry = delta + gamma * gae_lambda * (1-done[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


def clipped_policy_loss(log_probs, old_log_probs, advantages, clip_range):
    ratio = (log_probs-old_log_probs).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1-clip_range, 1+clip_range) * advantages
    return -torch.minimum(unclipped, clipped).mean()


class ActorCritic(nn.Module):
    def __init__(self, dimensions, actions, hidden):
        super().__init__()
        # Separate networks prevent value regression from dominating actor features.
        def mlp(outputs, output_gain):
            net = nn.Sequential(nn.Linear(dimensions, hidden), nn.Tanh(),
                                nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, outputs))
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, np.sqrt(2))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(net[-1].weight, output_gain)
            return net
        self.actor = mlp(actions, .01)
        self.critic = mlp(1, 1.)

    def forward(self, states):
        return Categorical(logits=self.actor(states)), self.critic(states).squeeze(-1)


class PPOAgent:
    def __init__(self, config, columns, mean, std, seed=23):
        self.config, self.columns, self.seed = config, list(columns), seed
        self.positions = tuple(config.positions)
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        if len(self.mean) != len(self.columns)+2 or np.any(self.std <= 0):
            raise ValueError('Scaler must match static features plus two position features')
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.network = ActorCritic(len(self.mean), len(self.positions), config.hidden)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.lr, eps=1e-5)
        self.updates = 0

    def encode(self, states):
        encoded = np.clip((np.asarray(states, np.float32)-self.mean)/self.std, -8, 8)
        if not np.isfinite(encoded).all():
            raise FloatingPointError('Non-finite observation')
        return encoded

    def probabilities(self, state):
        with torch.no_grad():
            distribution, _ = self.network(torch.from_numpy(self.encode(state)))
        return distribution.probs.numpy()

    def choose_action_eval(self, state):
        return int(self.probabilities(state).argmax())

    def update(self, states, actions, old_log_probs, advantages, returns, progress):
        cfg = self.config
        states = torch.as_tensor(states.reshape(-1, len(self.mean)))
        actions = torch.as_tensor(actions.reshape(-1), dtype=torch.long)
        old_log_probs = torch.as_tensor(old_log_probs.reshape(-1))
        advantages = torch.as_tensor(advantages.reshape(-1))
        advantages = (advantages-advantages.mean()) / (advantages.std(unbiased=False)+1e-8)
        returns = torch.as_tensor(returns.reshape(-1))
        lr = cfg.lr * (max(.1, 1-progress) if cfg.anneal_lr else 1.)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        metrics, stopped = [], False
        for _ in range(cfg.epochs):
            permutation = self.rng.permutation(len(actions))
            for offset in range(0, len(actions), cfg.batch_size):
                indices = permutation[offset:offset+cfg.batch_size]
                distribution, values = self.network(states[indices])
                log_probs = distribution.log_prob(actions[indices])
                log_ratio = log_probs-old_log_probs[indices]
                with torch.no_grad():
                    approx_kl = ((log_ratio.exp()-1)-log_ratio).mean()
                if cfg.target_kl and approx_kl > 1.5*cfg.target_kl:
                    stopped = True
                    break
                actor_loss = clipped_policy_loss(log_probs, old_log_probs[indices],
                                                 advantages[indices], cfg.clip_range)
                value_loss = .5 * (values-returns[indices]).square().mean()
                entropy = distribution.entropy().mean()
                loss = actor_loss + cfg.value_coef*value_loss - cfg.entropy_coef*entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite PPO loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError('Non-finite PPO gradient')
                self.optimizer.step()
                self.updates += 1
                metrics.append([actor_loss.item(), value_loss.item(), entropy.item(),
                                approx_kl.item(), ((log_ratio.exp()-1).abs()>cfg.clip_range).float().mean().item()])
            if stopped:
                break
        if not metrics:
            raise RuntimeError('No PPO update completed')
        result = dict(zip(['actor_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction'],
                          np.mean(metrics, axis=0).tolist()))
        with torch.no_grad():
            predicted = self.network.critic(states).squeeze(-1)
            variance = returns.var(unbiased=False)
            result['explained_variance'] = float(1-(returns-predicted).var(unbiased=False)/(variance+1e-8))
        return {**result, 'learning_rate': lr, 'kl_early_stop': int(stopped), 'updates': self.updates}

    def payload(self):
        return dict(config=asdict(self.config), columns=self.columns, mean=self.mean.tolist(),
                    std=self.std.tolist(), seed=self.seed, weights=self.network.state_dict(),
                    updates=self.updates)

    @classmethod
    def from_payload(cls, payload):
        agent = cls(config_from_dict(payload['config']), payload['columns'],
                    payload['mean'], payload['std'], payload['seed'])
        agent.network.load_state_dict(payload['weights'])
        agent.network.eval()
        agent.updates = payload['updates']
        return agent


class PPOEnsemble:
    def __init__(self, agents, aggregation='mean_probability'):
        if not agents or aggregation not in AGGREGATIONS:
            raise ValueError('Unknown aggregation or empty ensemble')
        self.agents, self.aggregation = agents, aggregation
        self.config, self.positions = agents[0].config, agents[0].positions
        self.columns = agents[0].columns
        if any(a.config != self.config or a.columns != self.columns for a in agents):
            raise ValueError('Ensemble configurations and feature orders must match')

    def choose_action_eval(self, state):
        probabilities = np.array([a.probabilities(state) for a in self.agents])
        if self.aggregation == 'mean_probability':
            return int(probabilities.mean(axis=0).argmax())
        votes = probabilities.argmax(axis=1)
        return int(votes[0]) if np.all(votes == votes[0]) else self.positions.index(0.)

    def save(self, path):
        torch.save(dict(format='trading_ppo_v1', aggregation=self.aggregation,
                        agents=[a.payload() for a in self.agents]), path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format') != 'trading_ppo_v1':
            raise ValueError('Not a PPO trading checkpoint')
        return cls([PPOAgent.from_payload(p) for p in payload['agents']], payload['aggregation'])


def fit(frame, cfg, seed, steps=CONFIRM_STEPS):
    """Synchronous independent episodes; each rollout is discarded after updating."""
    if steps < cfg.n_envs or steps % cfg.n_envs:
        raise ValueError('Training steps must be a positive multiple of n_envs')
    if len(frame) < 3:
        raise ValueError('Training requires at least three candles')
    torch.set_num_threads(1)
    columns = [c for c in frame if c.startswith('feature_')]
    mean = np.r_[frame[columns].mean().to_numpy(), [0., 0.]]
    std = np.r_[frame[columns].std().clip(lower=1e-6).to_numpy(), [1., 1.]]
    agent = PPOAgent(cfg, columns, mean, std, seed)
    environments = [TrainingWindowEnv(frame, cfg.positions, duration=min(720, len(frame)-1))
                    for _ in range(cfg.n_envs)]
    episodes = np.zeros(cfg.n_envs, dtype=int)

    def reset(index):
        reset_seed = seed + index + int(episodes[index])*cfg.n_envs
        # This TradingEnv release uses global NumPy RNG for random starts.
        np.random.seed(reset_seed)
        state, _ = environments[index].reset(seed=reset_seed)
        episodes[index] += 1
        return state

    logs, total = [], 0
    try:
        raw_states = np.stack([reset(i) for i in range(cfg.n_envs)])
        while total < steps:
            length = min(cfg.rollout_steps, (steps-total)//cfg.n_envs)
            shape = (length, cfg.n_envs)
            states = np.empty((*shape, len(mean)), np.float32)
            next_states = np.empty_like(states)
            actions = np.empty(shape, np.int64)
            rewards, values, log_probs, terminated, done = [np.empty(shape, np.float32) for _ in range(5)]
            raw_rewards = []
            for t in range(length):
                states[t] = agent.encode(raw_states)
                with torch.no_grad():
                    distribution, value = agent.network(torch.from_numpy(states[t]))
                    action = distribution.sample()
                    actions[t] = action.numpy()
                    values[t] = value.numpy()
                    log_probs[t] = distribution.log_prob(action).numpy()
                for i, env in enumerate(environments):
                    future, reward, terminal, truncated, _ = env.step(int(actions[t, i]))
                    # Bankruptcy returns zero in TradingEnv; give learning a real penalty.
                    shaped_reward = -1. if terminal else float(reward)
                    shaped_reward -= cfg.turnover_penalty * abs(cfg.positions[actions[t,i]]-float(raw_states[i,-2]))
                    rewards[t,i] = shaped_reward * cfg.reward_scale
                    terminated[t,i], done[t,i] = terminal, terminal or truncated
                    # Capture final observation before resetting for GAE bootstrapping.
                    next_states[t,i] = np.zeros(len(mean), np.float32) if terminal else agent.encode(future)
                    raw_rewards.append(float(reward))
                    raw_states[i] = reset(i) if terminal or truncated else future
            with torch.no_grad():
                next_values = agent.network.critic(torch.from_numpy(next_states.reshape(-1,len(mean))))
                next_values = next_values.numpy().reshape(shape)
            if not np.isfinite(rewards).all():
                raise FloatingPointError('Non-finite PPO reward')
            advantages, returns = compute_gae(rewards, values, next_values, terminated, done,
                                             cfg.gamma, cfg.gae_lambda)
            total += length*cfg.n_envs
            diagnostics = agent.update(states, actions, log_probs, advantages, returns, total/steps)
            logs.append(dict(steps=total, episodes=int(episodes.sum()-cfg.n_envs),
                             mean_raw_step_reward=float(np.mean(raw_rewards)), **diagnostics))
    finally:
        for env in environments:
            env.close()
    agent.network.eval()
    return agent, pd.DataFrame(logs)


def ranking(results):
    summary = results.groupby('config').agg(mean_log_return=('log_return','mean'),
        std_log_return=('log_return','std'), mean_return=('portfolio_return','mean'),
        worst_return=('portfolio_return','min'), mean_drawdown=('max_drawdown','mean'),
        mean_action_changes=('action_changes','mean'))
    summary['selection_score'] = summary.mean_log_return + .25*summary.mean_drawdown - .1*summary.std_log_return
    return summary.sort_values('selection_score', ascending=False)


def protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    pre_eval = load_data().loc[lambda x:x.index<EVAL_START]
    description = dict(configurations=[asdict(c) for c in CONFIGS],
        train_start=str(TRAIN_START), evaluation_start=str(EVAL_START), evaluation_end=str(EVAL_END),
        data_path=DATA_PATH.name,
        folds=[[str(start),str(end)] for start,end in FOLDS], screening_steps=SCREEN_STEPS,
        confirmation_steps=CONFIRM_STEPS, final_steps=CONFIRM_STEPS,
        screening_seeds=[23], confirmation_seeds=SEEDS, aggregations=list(AGGREGATIONS),
        finalists='top two screening configurations; also confirm baseline for fair comparison',
        score='mean(log return) - 0.25*mean(drawdown magnitude) - 0.10*std(log return)',
        evaluation_used_for_selection=False, reused_assignment_holdout=True,
        pre_eval_data_sha256=hashlib.sha256(pd.util.hash_pandas_object(pre_eval,index=True).values.tobytes()).hexdigest(),
        ppo_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment_source_sha256=hashlib.sha256((ROOT/'trading_tuning.py').read_bytes()).hexdigest(),
        data_source_sha256=hashlib.sha256((ROOT/'btc_data.py').read_bytes()).hexdigest())
    path = OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != description:
        # Round-trip handles tuple/list serialization in the comparison.
        if json.loads(path.read_text()) != json.loads(json.dumps(description)):
            raise ValueError('Protocol/source changed: preserve old artifacts and use a fresh OUT directory')
    path.write_text(json.dumps(description, indent=2))
    return description


def train_job(cfg_dict, seed, fold, steps, phase):
    cfg = config_from_dict(cfg_dict)
    frame = add_features(load_data().loc[lambda x:x.index<EVAL_START], cfg.features)
    boundary = EVAL_START if fold == -1 else FOLDS[fold][0]
    training = frame.loc[(frame.index>=TRAIN_START)&(frame.index<boundary)]
    agent, history = fit(training, cfg, seed, steps)
    path = OUT/phase/f'{cfg.name}_fold{fold}_seed{seed}'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.payload(), path.with_suffix('.pth'))
    history.to_csv(path.with_suffix('.training.csv'), index=False)
    row = dict(config=cfg.name, seed=seed, fold=fold, training_steps=steps, updates=agent.updates)
    if fold != -1:
        start, end = FOLDS[fold]
        validation = frame.loc[(frame.index>=start)&(frame.index<end)]
        assert training.index.max() < validation.index.min() and validation.index.max() < EVAL_START
        row.update(start=str(start), end=str(end), **rollout(agent, validation))
    path.with_suffix('.json').write_text(json.dumps(row, indent=2))
    return row


def run_jobs(configs, seeds, steps, phase, workers=3, folds=None):
    protocol()
    folds = list(range(len(FOLDS))) if folds is None else folds
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for cfg in configs:
            for fold in folds:
                for seed in seeds:
                    path = OUT/phase/f'{cfg.name}_fold{fold}_seed{seed}'
                    if all(path.with_suffix(s).exists() for s in ['.json','.pth','.training.csv']):
                        row = json.loads(path.with_suffix('.json').read_text())
                        if row['training_steps'] != steps:
                            raise ValueError('Cached trial budget differs')
                        rows.append(row)
                    else:
                        futures.append(pool.submit(train_job, asdict(cfg), seed, fold, steps, phase))
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(phase, row['config'], 'fold', row['fold'], 'seed', row['seed'],
                  'return', f"{row['portfolio_return']:.2%}" if 'portfolio_return' in row else 'fit complete', flush=True)
    expected = {(c.name, f, s) for c in configs for f in folds for s in seeds}
    keys = [(r['config'],r['fold'],r['seed']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Incomplete or duplicate trial grid')
    results = pd.DataFrame(rows).sort_values(['config','fold','seed'])
    if not np.isfinite(results.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError('Non-finite trial metrics')
    results.to_csv(OUT/f'{phase}_trials.csv', index=False)
    return results


def load_member(cfg, seed, fold, phase):
    path = OUT/phase/f'{cfg.name}_fold{fold}_seed{seed}.pth'
    return PPOAgent.from_payload(torch.load(path, map_location='cpu', weights_only=True))


def screen(workers=3):
    results = run_jobs(CONFIGS, [23], SCREEN_STEPS, 'screening', workers)
    ranks = ranking(results)
    ranks.to_csv(OUT/'screening_summary.csv')
    print(ranks.to_string(), flush=True)
    return ranks


def confirm(workers=3):
    screening = run_jobs(CONFIGS, [23], SCREEN_STEPS, 'screening', workers)
    ranks = ranking(screening)
    ranks.to_csv(OUT/'screening_summary.csv')
    finalists = set(ranks.index[:2]) | {'ppo_baseline'}
    configs = [c for c in CONFIGS if c.name in finalists]
    results = run_jobs(configs, SEEDS, CONFIRM_STEPS, 'confirmation', workers)
    ranking(results).to_csv(OUT/'confirmation_individual_summary.csv')
    raw = load_data().loc[lambda x:x.index<EVAL_START]
    rows, benchmarks = [], []
    for cfg in configs:
        frame = add_features(raw, cfg.features)
        for fold,(start,end) in enumerate(FOLDS):
            members = [load_member(cfg, seed, fold, 'confirmation') for seed in SEEDS]
            part = frame.loc[(frame.index>=start)&(frame.index<end)]
            for method in AGGREGATIONS:
                policy = PPOEnsemble(members, method)
                rows.append(dict(config=cfg.name+'::'+method, fold=fold, **rollout(policy,part)))
    frame = add_features(raw)
    for fold,(start,end) in enumerate(FOLDS):
        part = frame.loc[(frame.index>=start)&(frame.index<end)]
        for name, position in [('Cash',0.), ('Buy and hold',1.), ('Constant short',-1.)]:
            benchmarks.append(dict(config=name, fold=fold,
                **rollout(ConstantPolicy((-1.,0.,1.),position),part)))
    ensemble_results = pd.DataFrame(rows)
    ensemble_results.to_csv(OUT/'ensemble_validation.csv', index=False)
    pd.DataFrame(benchmarks).to_csv(OUT/'validation_benchmarks.csv',index=False)
    ranked = ranking(ensemble_results)
    ranked.to_csv(OUT/'ensemble_summary.csv')
    name, aggregation = ranked.index[0].split('::')
    selected = next(c for c in configs if c.name==name)
    selection = dict(config=asdict(selected), aggregation=aggregation, seeds=SEEDS,
        training_steps=CONFIRM_STEPS, evaluation_used_for_selection=False,
        reused_assignment_holdout=True,
        selection='highest fixed risk-adjusted score over four chronological validation folds')
    (OUT/'selected_configuration.json').write_text(json.dumps(selection,indent=2))
    print('FROZEN PPO SELECTION', name, aggregation, flush=True)
    print(ranked.to_string(),flush=True)
    return selection


def finalize(workers=3):
    protocol()
    selection = json.loads((OUT/'selected_configuration.json').read_text())
    cfg = config_from_dict(selection['config'])
    baseline_cfg = CONFIGS[0]
    configs = list({c.name:c for c in [cfg, baseline_cfg]}.values())
    run_jobs(configs, SEEDS, CONFIRM_STEPS, 'final_members', workers, folds=[-1])
    policy = PPOEnsemble([load_member(cfg,s,-1,'final_members') for s in SEEDS],selection['aggregation'])
    baseline = PPOEnsemble([load_member(baseline_cfg,s,-1,'final_members') for s in SEEDS])
    policy.save(OUT/'final_ppo_agent.pth')
    baseline.save(OUT/'baseline_ppo_agent.pth')
    # No more training or selection occurs after this point. The report reads eval data.
    from ppo_report import build_report
    build_report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['screen','confirm','final','all'])
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.phase == 'screen':
        screen(args.workers)
    elif args.phase == 'confirm':
        confirm(args.workers)
    elif args.phase == 'final':
        finalize(args.workers)
    else:
        confirm(args.workers)
        finalize(args.workers)


if __name__ == '__main__':
    main()
