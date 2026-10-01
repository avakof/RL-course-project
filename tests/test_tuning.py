"""Fast regression checks for tuning mechanics, using synthetic candles only.

Run with ``.venv-rl/bin/python tests/test_tuning.py`` from the project root.
These fixtures are never used for trading model selection or reported returns.
"""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trading_tuning as tuning


def synthetic_candles(length=720):
    rng = np.random.default_rng(314)
    close = 100 * np.exp(np.cumsum(rng.normal(0.00003, 0.002, length)))
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {"open": opening, "high": np.maximum(opening, close) * 1.001,
         "low": np.minimum(opening, close) * 0.999, "close": close,
         "volume": rng.lognormal(5, 0.3, length)},
        index=pd.date_range("2020-01-01", periods=length, freq="h", tz="UTC"),
    )


class FeatureAndTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.raw = synthetic_candles()
        cls.frame = tuning.add_features(cls.raw)
        cls.training = cls.frame.iloc[:360].copy()
        cls.validation = cls.frame.iloc[360:].copy()
        cls.config = tuning.Config("synthetic_test", hidden=8, batch=8)
        cls.agent, cls.history = tuning.fit(cls.training, cls.config, seed=11, steps=320)

    def test_features_ignore_future_candles(self):
        boundary = self.raw.index[500]
        perturbed = self.raw.copy()
        perturbed.loc[boundary:, ["open", "high", "low", "close"]] *= 20
        perturbed.loc[boundary:, "volume"] *= 100
        for kind in ["basic", "momentum"]:
            with self.subTest(features=kind):
                original = tuning.add_features(self.raw, kind)
                changed = tuning.add_features(perturbed, kind)
                pd.testing.assert_frame_equal(original.loc[original.index < boundary],
                                              changed.loc[changed.index < boundary])
                # Prefix processing must also give the identical historical features.
                pd.testing.assert_frame_equal(original.loc[original.index < boundary],
                                              tuning.add_features(self.raw.loc[self.raw.index < boundary], kind))

    def test_scaler_uses_training_partition(self):
        columns = self.agent.columns
        expected_mean = np.r_[self.training[columns].mean().to_numpy(), [0, 0]]
        expected_std = np.r_[self.training[columns].std().clip(lower=1e-6).to_numpy(), [1, 1]]
        np.testing.assert_allclose(self.agent.mean, expected_mean, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(self.agent.std, expected_std, rtol=1e-6, atol=1e-8)
        # Large validation features would expose accidental full-frame fitting.
        extreme_validation = self.validation.copy()
        extreme_validation.loc[:, columns] += 1000
        full_mean = pd.concat([self.training, extreme_validation])[columns].mean().to_numpy()
        self.assertFalse(np.allclose(self.agent.mean[:-2], full_mean))
        before = self.agent.mean.copy(), self.agent.std.copy()
        tuning.rollout(self.agent, extreme_validation)
        np.testing.assert_array_equal(self.agent.mean, before[0])
        np.testing.assert_array_equal(self.agent.std, before[1])

    def test_tiny_fit_and_full_validation_rollout(self):
        self.assertEqual(int(self.history.steps.iloc[-1]), 320)
        self.assertGreater(self.agent.updates, 0)
        self.assertTrue(np.isfinite(self.history.select_dtypes(include=[np.number])).all().all())
        metrics, trace = tuning.rollout(self.agent, self.validation, return_trace=True)
        self.assertEqual(metrics["steps"], len(self.validation) - 1)
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        self.assertTrue((trace.portfolio_value > 0).all())
        self.assertEqual(trace.timestamp.tolist(), self.validation.index.tolist())
        self.assertTrue(set(trace.position).issubset(set(self.config.positions)))
        self.assertAlmostEqual(metrics["log_return"], metrics["total_reward"], places=10)
        self.assertEqual(metrics, tuning.rollout(self.agent, self.validation))

    def test_ensemble_checkpoint_round_trip(self):
        second = tuning.TunedDQN(self.config, self.agent.columns, self.agent.mean, self.agent.std, seed=19)
        policy = tuning.EnsemblePolicy([self.agent, second])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.pth"
            policy.save(path)
            loaded = tuning.EnsemblePolicy.load(path)
        self.assertEqual(len(loaded.agents), 2)
        self.assertEqual(loaded.positions, policy.positions)
        self.assertEqual(loaded.config, policy.config)
        self.assertEqual(loaded.columns, policy.columns)
        for original, restored in zip(policy.agents, loaded.agents):
            np.testing.assert_array_equal(original.mean, restored.mean)
            np.testing.assert_array_equal(original.std, restored.std)
            for old, new in zip(original.network.parameters(), restored.network.parameters()):
                torch.testing.assert_close(old, new, rtol=0, atol=0)
            for online, target in zip(restored.network.parameters(), restored.target.parameters()):
                torch.testing.assert_close(online, target, rtol=0, atol=0)
                self.assertNotEqual(online.data_ptr(), target.data_ptr())
        for _, row in self.validation.iloc[::30].iterrows():
            state = np.r_[row[self.agent.columns].to_numpy(), [0, 0]].astype(np.float32)
            expected = int(np.mean([agent.q_values(state) for agent in policy.agents], axis=0).argmax())
            self.assertEqual(policy.choose_action_eval(state), expected)
            self.assertEqual(loaded.choose_action_eval(state), expected)
        self.assertEqual(tuning.rollout(policy, self.validation), tuning.rollout(loaded, self.validation))


class ReplayAndTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ring_buffer_wrap_and_sample_alignment(self):
        replay = tuning.Replay(size=3, dimensions=2, rng=np.random.default_rng(2))
        for value in range(7):
            state = np.array([value, value + 0.5], dtype=np.float32)
            replay.append(state, value, value * 10, state + 1, value % 2)
            state[:] = -999  # Stored transitions must own their state values.
        self.assertEqual(replay.count, 3)
        self.assertEqual(replay.cursor, 1)
        states, actions, rewards, next_states, terminals = replay.sample(3)
        self.assertEqual(set(actions.tolist()), {4, 5, 6})
        self.assertEqual(actions.dtype, torch.int64)
        self.assertEqual(states.dtype, torch.float32)
        np.testing.assert_array_equal(states[:, 0].numpy(), actions.numpy())
        np.testing.assert_array_equal(states[:, 1].numpy(), actions.numpy() + 0.5)
        torch.testing.assert_close(next_states, states + 1)
        np.testing.assert_array_equal(rewards.numpy(), actions.numpy() * 10)
        np.testing.assert_array_equal(terminals.numpy(), actions.numpy() % 2)

    @staticmethod
    def populated_agent(tau):
        config = tuning.Config("target_test", hidden=8, batch=4, lr=0.01, tau=tau)
        agent = tuning.TunedDQN(config, ["feature_test"], [0, 0, 0], [1, 1, 1], seed=7)
        for value in range(8):
            state = np.array([value / 10, 0, 0], dtype=np.float32)
            agent.replay.append(state, value % len(config.positions), 0.3 + value / 10,
                                state + np.array([0.01, 0, 0], dtype=np.float32), value % 2)
        return agent

    def test_soft_target_has_separate_weights_and_no_gradients(self):
        agent = self.populated_agent(tau=0.2)
        before = [parameter.detach().clone() for parameter in agent.target.parameters()]
        for online, target in zip(agent.network.parameters(), agent.target.parameters()):
            self.assertNotEqual(online.data_ptr(), target.data_ptr())
            self.assertFalse(target.requires_grad)
        loss = agent.learn()
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(agent.updates, 1)
        self.assertTrue(any(not torch.equal(old, online) for old, online in zip(before, agent.network.parameters())))
        for old, online, target in zip(before, agent.network.parameters(), agent.target.parameters()):
            torch.testing.assert_close(target, old + 0.2 * (online.detach() - old))
            self.assertIsNone(target.grad)

    def test_hard_target_waits_until_sync_interval(self):
        agent = self.populated_agent(tau=0)
        before = [parameter.detach().clone() for parameter in agent.target.parameters()]
        agent.learn()
        for old, target in zip(before, agent.target.parameters()):
            torch.testing.assert_close(old, target, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(online, target)
                            for online, target in zip(agent.network.parameters(), agent.target.parameters())))
        agent.updates = 249
        agent.learn()
        self.assertEqual(agent.updates, 250)
        for online, target in zip(agent.network.parameters(), agent.target.parameters()):
            torch.testing.assert_close(online, target, rtol=0, atol=0)


class ValidatedAggregationTests(unittest.TestCase):
    positions = (-1., 0., 1.)
    state = np.zeros(3, dtype=np.float32)

    @classmethod
    def fixed_q_members(cls, rows):
        config = tuning.Config("aggregation_test", hidden=8, positions=cls.positions)
        members = []
        for row in rows:
            values = np.asarray(row, dtype=np.float32)
            members.append(SimpleNamespace(
                positions=cls.positions, config=config, columns=["feature_test"],
                q_values=lambda state, values=values: values.copy(),
            ))
        return members

    def policy_with_votes(self, votes, aggregation):
        rows = np.eye(len(self.positions), dtype=np.float32)[votes]
        return tuning.ValidatedEnsemblePolicy(self.fixed_q_members(rows), aggregation)

    def test_majority_requires_strictly_more_than_half(self):
        cases = [
            ([0, 0, 2], 0),             # Two of three votes win.
            ([2, 2, 2, 0], 2),          # Three of four votes win.
            ([0, 0, 2, 2], 1),          # An even tie falls back to cash.
            ([0, 1, 2], 1),             # A three-way tie falls back to cash.
            ([0, 0, 2, 2, 1], 1),       # A plurality is insufficient.
            ([0, 0, 0, 2, 2, 1], 1),    # Exactly half is insufficient.
            ([2], 2),                   # One member supplies a strict majority.
        ]
        for votes, expected_action in cases:
            with self.subTest(votes=votes):
                policy = self.policy_with_votes(votes, "majority")
                self.assertEqual(policy.choose_action_eval(self.state), expected_action)

    def test_unanimous_requires_every_member_to_agree(self):
        cases = [([2, 2, 2], 2), ([0, 0, 0], 0), ([1, 1, 1], 1),
                 ([0, 0, 2], 1), ([2, 2, 2, 0], 1), ([0, 2], 1), ([2], 2)]
        for votes, expected_action in cases:
            with self.subTest(votes=votes):
                policy = self.policy_with_votes(votes, "unanimous")
                self.assertEqual(policy.choose_action_eval(self.state), expected_action)

    def test_normalized_advantage_ignores_member_scale_and_offset(self):
        # The first member has a much larger Q scale, but two members favor long.
        rows = np.array([[1000., 0., 0.], [0., 0., 1.], [0., 0., 1.]])
        members = self.fixed_q_members(rows)
        self.assertEqual(tuning.ValidatedEnsemblePolicy(members, "mean_q").choose_action_eval(self.state), 0)
        normalized = tuning.ValidatedEnsemblePolicy(members, "normalized_advantage")
        self.assertEqual(normalized.choose_action_eval(self.state), 2)
        transformed = rows * np.array([0.1, 3., 5.])[:, None] + np.array([7., -100., 200.])[:, None]
        changed = tuning.ValidatedEnsemblePolicy(self.fixed_q_members(transformed), "normalized_advantage")
        self.assertEqual(changed.choose_action_eval(self.state), 2)
        # A member with no action preference contributes zero normalized advantage.
        flat_member = self.fixed_q_members([[4., 4., 4.]])[0]
        with np.errstate(divide="raise", invalid="raise"):
            with_flat = tuning.ValidatedEnsemblePolicy(members + [flat_member], "normalized_advantage")
            self.assertEqual(with_flat.choose_action_eval(self.state), 2)

    def test_checkpoint_preserves_selected_aggregation(self):
        config = tuning.Config("aggregation_test", hidden=8, positions=self.positions)
        members = []
        for seed, row in enumerate([[1000., 0., 0.], [0., 0., 1.], [0., 0., 1.]]):
            agent = tuning.TunedDQN(config, ["feature_test"], [0, 0, 0], [1, 1, 1], seed=seed)
            with torch.no_grad():
                for parameter in agent.network.parameters():
                    parameter.zero_()
                agent.network[-1].bias.copy_(torch.tensor(row))
            members.append(agent)
        expected_actions = {"mean_q": 0, "majority": 2, "unanimous": 1, "normalized_advantage": 2}
        with tempfile.TemporaryDirectory() as directory:
            for method, expected_action in expected_actions.items():
                with self.subTest(aggregation=method):
                    policy = tuning.ValidatedEnsemblePolicy(members, method)
                    path = Path(directory) / f"{method}.pth"
                    policy.save(path)
                    restored = tuning.ValidatedEnsemblePolicy.load(path)
                    self.assertEqual(restored.aggregation, method)
                    self.assertEqual(restored.positions, self.positions)
                    self.assertEqual(restored.config, config)
                    self.assertEqual(restored.columns, ["feature_test"])
                    for state in [self.state, np.array([2., -1., 0.5], dtype=np.float32)]:
                        self.assertEqual(policy.choose_action_eval(state), expected_action)
                        self.assertEqual(restored.choose_action_eval(state), expected_action)
            legacy_path = Path(directory) / "legacy_mean_q.pth"
            tuning.EnsemblePolicy(members).save(legacy_path)
            legacy = tuning.ValidatedEnsemblePolicy.load(legacy_path)
            self.assertEqual(legacy.aggregation, "mean_q")
            self.assertEqual(legacy.choose_action_eval(self.state), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
