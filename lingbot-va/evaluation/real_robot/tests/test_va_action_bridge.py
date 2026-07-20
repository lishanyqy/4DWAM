#!/usr/bin/env python3
"""Unit tests for VA action expand / re-anchor / keyframe schedule (no ROS)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CLIENT_ROOT = Path(__file__).resolve().parents[1] / 'client'
sys.path.insert(0, str(CLIENT_ROOT))

from deploy.utils.rotation import (  # noqa: E402
    add_init_pose,
    eef14_to_eef16,
    eef16_to_eef14,
)
from deploy.utils.va_action_bridge import (  # noqa: E402
    VAActionBridge,
    count_keyframes,
    expand_action_chunk,
)


class TestKeyframeSchedule(unittest.TestCase):
    def test_count_first_and_later_chunks(self):
        # F=2, H=16, first start_idx=1 -> 4 keyframes; later start_idx=0 -> 8
        self.assertEqual(count_keyframes(2, 16, start_idx=1, keyframe_divisor=4), 4)
        self.assertEqual(count_keyframes(2, 16, start_idx=0, keyframe_divisor=4), 8)

    def test_expand_keyframe_flags(self):
        # 16D layout: xyz(3)+quat_xyzw(4)+grip(1) per arm
        action = np.zeros((16, 2, 16), dtype=np.float32)
        action[6, :, :] = 1.0   # left quat w
        action[14, :, :] = 1.0  # right quat w
        action[7, :, :] = 1.0   # left grip
        action[15, :, :] = 1.0  # right grip

        init14 = np.array(
            [0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, -0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        init16 = eef14_to_eef16(init14)

        first_steps = expand_action_chunk(
            action,
            init16,
            first_chunk=True,
            first_chunk_start_idx=1,
            keyframe_divisor=4,
        )
        self.assertEqual(len(first_steps), 16)
        self.assertEqual(sum(1 for step in first_steps if step.is_keyframe), 4)

        later_steps = expand_action_chunk(
            action,
            init16,
            first_chunk=False,
            first_chunk_start_idx=1,
            keyframe_divisor=4,
        )
        self.assertEqual(len(later_steps), 32)
        self.assertEqual(sum(1 for step in later_steps if step.is_keyframe), 8)


class TestReanchor(unittest.TestCase):
    def test_add_init_pose_translation(self):
        init14 = np.array(
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
            dtype=np.float64,
        )
        init16 = eef14_to_eef16(init14)
        relative = np.zeros(16, dtype=np.float64)
        relative[0:3] = [0.1, 0.0, 0.0]
        relative[3:7] = [0.0, 0.0, 0.0, 1.0]
        relative[7] = 1.0
        relative[8:11] = [0.0, 0.2, 0.0]
        relative[11:15] = [0.0, 0.0, 0.0, 1.0]
        relative[15] = 0.2

        absolute = add_init_pose(relative, init16)
        abs14 = eef16_to_eef14(absolute)
        np.testing.assert_allclose(abs14[0:3], [1.1, 2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(abs14[7:10], [0.0, 0.2, 0.0], atol=1e-6)
        # gripper is absolute from relative, not added
        self.assertAlmostEqual(float(abs14[6]), 1.0, places=5)
        self.assertAlmostEqual(float(abs14[13]), 0.2, places=5)

    def test_bridge_stateful_start_idx(self):
        action = np.zeros((16, 2, 16), dtype=np.float32)
        action[6, :, :] = 1.0
        action[14, :, :] = 1.0
        action[7, :, :] = 1.0
        action[15, :, :] = 1.0

        bridge = VAActionBridge(first_chunk_start_idx=1, keyframe_divisor=4)
        bridge.reset(
            np.array(
                [0, 0, 0.2, 0, 0, 0, 1, 0, 0, 0.2, 0, 0, 0, 1],
                dtype=np.float64,
            )
        )
        first = bridge.expand(action)
        second = bridge.expand(action)
        self.assertEqual(len(first), 16)
        self.assertEqual(len(second), 32)
        self.assertAlmostEqual(float(first[0].absolute_eef14_raw[6]), 5.0, places=5)


class TestRoundTrip14_16(unittest.TestCase):
    def test_eef14_16_roundtrip(self):
        eef14 = np.array(
            [0.1, -0.2, 0.3, 0.1, -0.1, 0.2, 0.7, 0.05, 0.0, 0.25, -0.05, 0.1, 0.0, 0.3],
            dtype=np.float64,
        )
        back = eef16_to_eef14(eef14_to_eef16(eef14))
        np.testing.assert_allclose(back[:3], eef14[:3], atol=1e-6)
        np.testing.assert_allclose(back[3:6], eef14[3:6], atol=1e-5)
        np.testing.assert_allclose(back[6], eef14[6], atol=1e-6)
        np.testing.assert_allclose(back[7:10], eef14[7:10], atol=1e-6)
        np.testing.assert_allclose(back[10:13], eef14[10:13], atol=1e-5)
        np.testing.assert_allclose(back[13], eef14[13], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
