"""Synthetic checks of PPO math, episode boundaries, training and persistence."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ppo_trading as ppo


def candles(length=420):
    rng = np.random.default_rng(123)
    close = 100*np.exp(np.cumsum(rng.normal(0, .002, length)))
    opening = np.r_[close[0],close[:-1]]
    return pd.DataFrame(dict(open=opening, close=close, high=np.maximum(opening,close)*1.001,
        low=np.minimum(opening,close)*.999, volume=rng.uniform(100,1000,length)),
        index=pd.date_range('2020-01-01', periods=length, freq='h', tz='UTC'))


class PPOMathTests(unittest.TestCase):
    def test_clipped_objective_and_gradients_both_advantage_signs(self):
        # High ratio with positive advantage and low ratio with negative advantage
        # must stop increasing the objective beyond the clipping boundary.
        ratios = torch.tensor([1.5, .5, .5, 1.5])
        log_probs = ratios.log().requires_grad_()
        advantages = torch.tensor([1., -1., 1., -1.])
        loss = ppo.clipped_policy_loss(log_probs, torch.zeros(4), advantages, .2)
        self.assertAlmostEqual(loss.item(), .15, places=6)
        loss.backward()
        torch.testing.assert_close(log_probs.grad, torch.tensor([0.,0.,-.125,.375]))

    def test_gae_terminal_truncation_and_normal_transition(self):
        rewards = np.array([[1.,1.,1.],[100.,100.,100.]], np.float32)
        values = np.full((2,3),.5,np.float32)
        next_values = np.full((2,3),2.,np.float32)
        terminated = np.array([[1,0,0],[0,0,0]],np.float32)
        done = np.array([[1,1,0],[0,0,0]],np.float32)
        advantages, returns = ppo.compute_gae(rewards,values,next_values,terminated,done,.9,.8)
        # Both terminated and truncated transitions stop GAE from using reset rewards.
        np.testing.assert_allclose(advantages[0], [.5,2.3,2.3+.72*101.3],rtol=1e-6)
        # An incomplete rollout still bootstraps the next observation.
        np.testing.assert_allclose(advantages[1],[101.3]*3,rtol=1e-6)
        np.testing.assert_allclose(returns,advantages+values)

    def test_gae_rejects_wrong_shapes(self):
        with self.assertRaises(ValueError):
            ppo.compute_gae(*[np.zeros(3)]*5,.99,.95)


class PPOTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.frame = ppo.add_features(candles())
        cls.training, cls.validation = cls.frame.iloc[:160], cls.frame.iloc[160:]
        cls.cfg = ppo.PPOConfig('synthetic',hidden=8,n_envs=2,rollout_steps=32,
                                batch_size=32,epochs=2)
        cls.agent, cls.history = ppo.fit(cls.training,cls.cfg,11,steps=392)

    def test_partial_rollout_and_finite_actor_critic_updates(self):
        self.assertEqual(int(self.history.steps.iloc[-1]),392)
        self.assertGreater(self.agent.updates,0)
        self.assertTrue(np.isfinite(self.history.to_numpy()).all())
        original = ppo.PPOAgent(self.cfg,self.agent.columns,self.agent.mean,self.agent.std,11)
        for kind in ['actor','critic']:
            before = list(getattr(original.network,kind).parameters())
            after = list(getattr(self.agent.network,kind).parameters())
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,after)))
        self.assertGreater(int(self.history.episodes.iloc[-1]),0)

    def test_training_is_reproducible_with_explicit_environment_resets(self):
        agent, history = ppo.fit(self.training,self.cfg,11,steps=392)
        pd.testing.assert_frame_equal(history,self.history)
        for first,second in zip(agent.network.parameters(),self.agent.network.parameters()):
            torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_scaling_uses_only_training_data_and_remains_frozen(self):
        np.testing.assert_allclose(self.agent.mean[:-2],self.training[self.agent.columns].mean(),rtol=1e-6)
        np.testing.assert_allclose(self.agent.std[:-2],self.training[self.agent.columns].std().clip(lower=1e-6),rtol=1e-6)
        before = self.agent.mean.copy(),self.agent.std.copy()
        ppo.rollout(self.agent,self.validation)
        np.testing.assert_array_equal(before[0],self.agent.mean)
        np.testing.assert_array_equal(before[1],self.agent.std)

    def test_checkpoint_reproduces_probabilities_and_entire_backtest(self):
        state = np.r_[self.validation.iloc[0][self.agent.columns],[0.,0.]].astype(np.float32)
        self.assertAlmostEqual(self.agent.probabilities(state).sum(),1.,places=6)
        for aggregation in ppo.AGGREGATIONS:
            policy = ppo.PPOEnsemble([self.agent],aggregation)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'ppo.pth'
                policy.save(path)
                restored=ppo.PPOEnsemble.load(path)
            self.assertEqual(restored.aggregation,aggregation)
            np.testing.assert_array_equal(restored.agents[0].probabilities(state),self.agent.probabilities(state))
            self.assertEqual(ppo.rollout(policy,self.validation),ppo.rollout(restored,self.validation))
        metrics,trace=ppo.rollout(self.agent,self.validation,return_trace=True)
        self.assertEqual(metrics['steps'],len(self.validation)-1)
        self.assertAlmostEqual(metrics['portfolio_return'],trace.portfolio_value.iloc[-1]/1000-1)
        self.assertAlmostEqual(metrics['log_return'],metrics['total_reward'],places=10)

    def test_unanimous_disagreement_holds_cash(self):
        class Member:
            config = self.cfg
            positions = self.cfg.positions
            columns = self.agent.columns
            def __init__(self,action): self.action=action
            def probabilities(self,state): return np.eye(len(self.positions))[self.action]
        agreed = ppo.PPOEnsemble([Member(3),Member(3)],'unanimous')
        disagreed = ppo.PPOEnsemble([Member(0),Member(3)],'unanimous')
        self.assertEqual(agreed.choose_action_eval(None),3)
        self.assertEqual(disagreed.choose_action_eval(None),self.cfg.positions.index(0.))

    def test_training_budget_validation(self):
        with self.assertRaises(ValueError): ppo.fit(self.training,self.cfg,11,steps=3)

    def test_nine_exposures_map_to_environment_and_actor(self):
        expected = (-1., -.75, -.5, -.25, 0., .25, .5, .75, 1.)
        self.assertTrue(all(cfg.positions == expected for cfg in ppo.CONFIGS))
        env = ppo.make_env(self.validation, self.cfg.positions)
        try:
            self.assertEqual(env.action_space.n, 9)
            for index, exposure in enumerate(expected):
                state, _ = env.reset(seed=23)
                self.assertEqual(self.agent.probabilities(state).shape, (9,))
                _, reward, _, _, info = env.step(index)
                self.assertEqual(info['position'], exposure)
                self.assertTrue(np.isfinite(reward))
                if exposure == 0:
                    self.assertEqual(index, 4)
                    self.assertEqual(reward, 0.)
        finally:
            env.close()

    def test_features_and_training_windows_do_not_cross_data_gaps(self):
        raw = candles(1600).drop(candles(1600).index[700:702])
        featured = ppo.add_features(raw)
        first_after_gap = raw.index[700]
        self.assertNotIn(first_after_gap, featured.index)
        self.assertIn(first_after_gap + pd.Timedelta(hours=168), featured.index)
        pd.testing.assert_frame_equal(featured.loc[first_after_gap:],
                                      ppo.add_features(raw.loc[first_after_gap:]))
        env = ppo.TrainingWindowEnv(featured, ppo.POSITIONS, 100)
        try:
            for seed in range(20):
                np.random.seed(seed)
                env.reset(seed=seed)
                times = env.unwrapped.df.index
                self.assertEqual(len(times), 100)
                self.assertTrue((times[1:] - times[:-1] == pd.Timedelta(hours=1)).all())
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
