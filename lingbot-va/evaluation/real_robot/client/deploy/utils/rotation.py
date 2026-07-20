#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEF / gripper / pose helpers for LIFT2 + VA.

ROS / openpi-on-LIFT2 side uses absolute 14D:
  [L_xyz(3), L_rpy(3), L_grip, R_xyz(3), R_rpy(3), R_grip]
  grip in [0, 1] for policy-facing state; raw robot grip is typically [0, 5].

VA used-channel layout is 16D (after server postprocess):
  [L_xyz(3), L_quat(4), L_grip, R_xyz(3), R_quat(4), R_grip]

Quaternion convention (must match **training supervision**, not docs alone)
----------------------------------------------------------------------
Training path in ``lerobot_latent_dataset``:

1. ``action14_to_action16`` uses ``utils.geometry.euler2quat`` (documented wxyz).
2. ``get_relative_pose`` then uses ``scipy.spatial.transform.Rotation.from_quat`` /
   ``as_quat`` on those 4 channels. SciPy's API is **xyzw**.
3. Therefore the tensors the model is **trained to predict** (after segment-relative)
   are composed / stored with SciPy **xyzw** semantics.

This client therefore treats all 16D quaternions as **xyzw** and uses SciPy for
re-anchor composition, matching the inverse of training's:

    relative_rot = first_rot.inv() * rot
    => rot = first_rot * relative_rot
    => abs_R = init_R * relative_R
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

GRIPPER_MIN = 0.0
GRIPPER_MAX = 5.0


def normalize_gripper(gripper_raw, gripper_max: float = GRIPPER_MAX):
    return np.clip(gripper_raw, GRIPPER_MIN, gripper_max) / float(gripper_max)


def denormalize_gripper(gripper_norm, gripper_max: float = GRIPPER_MAX):
    return np.clip(gripper_norm, 0.0, 1.0) * float(gripper_max)


def pose_to_eef(arm_left_pose, arm_right_pose, gripper_max: float = GRIPPER_MAX):
    """Extract absolute 14D EEF from ROS PosCmd messages (grip normalized)."""
    left_xyz = np.array(
        [arm_left_pose.x, arm_left_pose.y, arm_left_pose.z],
        dtype=np.float64,
    )
    left_rpy = np.array(
        [arm_left_pose.roll, arm_left_pose.pitch, arm_left_pose.yaw],
        dtype=np.float64,
    )
    left_gripper = normalize_gripper(arm_left_pose.gripper, gripper_max=gripper_max)

    right_xyz = np.array(
        [arm_right_pose.x, arm_right_pose.y, arm_right_pose.z],
        dtype=np.float64,
    )
    right_rpy = np.array(
        [arm_right_pose.roll, arm_right_pose.pitch, arm_right_pose.yaw],
        dtype=np.float64,
    )
    right_gripper = normalize_gripper(arm_right_pose.gripper, gripper_max=gripper_max)

    return np.concatenate([
        left_xyz,
        left_rpy,
        [left_gripper],
        right_xyz,
        right_rpy,
        [right_gripper],
    ]).astype(np.float64)


def eef_to_pose(eef):
    eef = np.asarray(eef, dtype=np.float64).reshape(14)
    return eef[:7].copy(), eef[7:14].copy()


def euler_xyz_to_quat_xyzw(roll, pitch, yaw):
    """Euler xyz (radians) -> quaternion **xyzw** (SciPy / training relative pose)."""
    return Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()


def quat_xyzw_to_euler_xyz(quat_xyzw):
    """Quaternion **xyzw** -> Euler xyz (radians)."""
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        raise ValueError(f'Quaternion has near-zero norm: {quat}')
    quat = quat / norm
    return Rotation.from_quat(quat).as_euler('xyz', degrees=False)


def eef14_to_eef16(eef14):
    """Absolute 14D euler EEF -> 16D xyz+quat_xyzw+grip."""
    eef14 = np.asarray(eef14, dtype=np.float64).reshape(14)
    left_quat = euler_xyz_to_quat_xyzw(eef14[3], eef14[4], eef14[5])
    right_quat = euler_xyz_to_quat_xyzw(eef14[10], eef14[11], eef14[12])
    return np.concatenate([
        eef14[0:3],
        left_quat,
        eef14[6:7],
        eef14[7:10],
        right_quat,
        eef14[13:14],
    ]).astype(np.float64)


def eef16_to_eef14(eef16):
    """16D xyz+quat_xyzw+grip -> absolute 14D euler EEF (grip unchanged)."""
    eef16 = np.asarray(eef16, dtype=np.float64).reshape(16)
    left_rpy = quat_xyzw_to_euler_xyz(eef16[3:7])
    right_rpy = quat_xyzw_to_euler_xyz(eef16[11:15])
    return np.concatenate([
        eef16[0:3],
        left_rpy,
        eef16[7:8],
        eef16[8:11],
        right_rpy,
        eef16[15:16],
    ]).astype(np.float64)


def _add_eef_pose_8d(relative_pose, init_pose):
    """Compose one arm: relative 8D (xyz+quat_xyzw+grip) onto init 8D absolute.

    Matches inverse of training ``get_relative_pose``:
      relative_R = first_R.inv() * R  =>  R = first_R * relative_R
    Gripper is absolute in the VA representation (not added to init grip).
    """
    relative_pose = np.asarray(relative_pose, dtype=np.float64).reshape(8)
    init_pose = np.asarray(init_pose, dtype=np.float64).reshape(8)

    relative_rotation = Rotation.from_quat(relative_pose[3:7])
    init_rotation = Rotation.from_quat(init_pose[3:7])
    out_rotation = (init_rotation * relative_rotation).as_quat().reshape(-1)
    out_translation = relative_pose[:3] + init_pose[:3]
    out_gripper = relative_pose[7:8]
    return np.concatenate([out_translation, out_rotation, out_gripper])


def add_init_pose(relative_eef16, init_eef16):
    """Re-anchor segment-relative 16D action to absolute world EEF."""
    relative_eef16 = np.asarray(relative_eef16, dtype=np.float64).reshape(16)
    init_eef16 = np.asarray(init_eef16, dtype=np.float64).reshape(16)
    left = _add_eef_pose_8d(relative_eef16[:8], init_eef16[:8])
    right = _add_eef_pose_8d(relative_eef16[8:], init_eef16[8:])
    absolute = np.concatenate([left, right])

    left_quat = absolute[3:7]
    right_quat = absolute[11:15]
    left_norm = np.linalg.norm(left_quat)
    right_norm = np.linalg.norm(right_quat)
    if left_norm < 1e-8 or right_norm < 1e-8:
        raise ValueError('Re-anchored quaternion has near-zero norm')
    absolute[3:7] = left_quat / left_norm
    absolute[11:15] = right_quat / right_norm
    return absolute.astype(np.float64)
