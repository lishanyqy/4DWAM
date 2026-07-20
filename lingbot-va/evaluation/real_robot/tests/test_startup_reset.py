#!/usr/bin/env python3
"""Tests for the real-robot startup reset motion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CLIENT_ROOT = Path(__file__).resolve().parents[1] / 'client'
sys.path.insert(0, str(CLIENT_ROOT))

from deploy.client_lift2_va import move_to_init_pose, parse_args  # noqa: E402


def create_pose(values):
    return SimpleNamespace(
        x=values[0],
        y=values[1],
        z=values[2],
        roll=values[3],
        pitch=values[4],
        yaw=values[5],
        gripper=values[6],
    )


class FakeRosOperator:
    def __init__(self, left_pose, right_pose):
        self.left_pose = left_pose
        self.right_pose = right_pose
        self.published_commands = []

    def get_latest_arm_poses(self):
        return self.left_pose, self.right_pose

    def eef_arm_publish(self, left_command, right_command):
        self.published_commands.append((left_command, right_command))


class TestStartupReset(unittest.TestCase):
    def test_interpolates_from_current_pose_and_denormalizes_gripper(self):
        ros_operator = FakeRosOperator(
            create_pose([0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 2.5]),
            create_pose([-0.2, -0.3, 0.5, -0.1, -0.2, -0.3, 5.0]),
        )

        move_to_init_pose(
            ros_operator=ros_operator,
            left_init_pose=[0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0],
            right_init_pose=[0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
            duration_seconds=1.0,
            publish_rate=2.0,
            gripper_raw_max=5.0,
            log_fn=lambda *_: None,
            sleep_fn=lambda *_: None,
        )

        self.assertEqual(len(ros_operator.published_commands), 3)
        first_left, first_right = ros_operator.published_commands[0]
        final_left, final_right = ros_operator.published_commands[-1]
        np.testing.assert_allclose(first_left, [0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 2.5])
        np.testing.assert_allclose(first_right, [-0.2, -0.3, 0.5, -0.1, -0.2, -0.3, 5.0])
        np.testing.assert_allclose(final_left, [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 5.0])
        np.testing.assert_allclose(final_right, [0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.0])

    def test_real_robot_profile_enables_reset_by_default(self):
        args = parse_args([])

        self.assertTrue(args.auto_init)
        self.assertEqual(args.init_duration, 3.0)
        self.assertEqual(args.left_init_pose, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(args.right_init_pose, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def test_no_auto_init_overrides_profile(self):
        args = parse_args(['--no_auto_init'])

        self.assertFalse(args.auto_init)


if __name__ == '__main__':
    unittest.main()
