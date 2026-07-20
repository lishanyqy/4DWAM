#!/usr/bin/env python3
"""Dedicated keyframe schedule tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CLIENT_ROOT = Path(__file__).resolve().parents[1] / 'client'
sys.path.insert(0, str(CLIENT_ROOT))

from deploy.client_lift2_va import run_mock_episode  # noqa: E402
from deploy.utils.va_action_bridge import count_keyframes  # noqa: E402


class FakePolicy:
    def __init__(self, actions):
        self.actions = actions
        self.events = []
        self.reset_prompts = []
        self.infer_observations = []
        self.kv_observations = []
        self.kv_states = []

    def reset(self, prompt=None):
        self.events.append('reset')
        self.reset_prompts.append(prompt)
        return {}

    def infer_action(self, observation):
        self.events.append('infer')
        self.infer_observations.append(observation)
        action = self.actions[len(self.infer_observations) - 1]
        return {'action': action}

    def compute_kv_cache(self, observations, state):
        self.events.append('compute_kv_cache')
        self.kv_observations.append(observations)
        self.kv_states.append(state)
        return {}


class TestKeyframeDivisor(unittest.TestCase):
    def test_requires_divisible_h(self):
        with self.assertRaises(ValueError):
            count_keyframes(2, 15, start_idx=0, keyframe_divisor=4)

    def test_robotwin_defaults(self):
        self.assertEqual(count_keyframes(2, 16, 1, 4), 4)
        self.assertEqual(count_keyframes(2, 16, 0, 4), 8)


class TestMockEpisodeFlow(unittest.TestCase):
    def test_mock_runs_infer_execute_kv_infer_sequence(self):
        action = np.zeros((16, 2, 16), dtype=np.float32)
        action[6, :, :] = 1.0
        action[14, :, :] = 1.0
        action[7, :, :] = 0.5
        action[15, :, :] = 0.5

        policy = FakePolicy([action, action.copy()])
        args = SimpleNamespace(
            language_instruction='mock task',
            camera_mode='tshape',
            first_chunk_start_idx=1,
            keyframe_divisor=4,
            gripper_raw_max=5.0,
            bridge_mode='segment_relative_va',
            dry_run_mock_chunks=2,
            verbose=False,
        )

        summaries = run_mock_episode(args, policy, log_fn=lambda *_: None)

        self.assertEqual(policy.reset_prompts, ['mock task'])
        self.assertEqual(
            policy.events,
            ['reset', 'infer', 'compute_kv_cache', 'infer', 'compute_kv_cache'],
        )
        self.assertEqual(len(policy.infer_observations), 2)
        self.assertEqual(len(policy.kv_observations), 2)
        self.assertEqual(
            [summary['executed_steps'] for summary in summaries],
            [16, 32],
        )
        self.assertEqual(
            [summary['keyframe_count'] for summary in summaries],
            [4, 8],
        )
        self.assertEqual(
            policy.infer_observations[0]['observation.images.cam_high'].shape,
            (256, 320, 3),
        )
        self.assertEqual(
            len(policy.kv_observations[0]),
            4,
        )
        self.assertEqual(
            len(policy.kv_observations[1]),
            8,
        )
        np.testing.assert_array_equal(policy.kv_states[0], action)
        np.testing.assert_array_equal(policy.kv_states[1], action)


if __name__ == '__main__':
    unittest.main()
