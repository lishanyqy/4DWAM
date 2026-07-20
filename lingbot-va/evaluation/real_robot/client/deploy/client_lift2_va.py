#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LIFT2 real-robot client for LingBot-VA / 4DWAM.

ROS I/O aligns with openpi-on-LIFT2 (absolute 14D EEF PosCmd).
Server protocol aligns with robotwin VA (reset / infer / compute_kv_cache).

Default profile: client/launch_profiles.yaml -> lift2_va_default
  host/port: 7777, publish_rate: 60
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLIENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(CLIENT_ROOT))

from dwam_client.websocket_va_policy import WebsocketVAPolicy  # noqa: E402
from deploy.utils.rotation import denormalize_gripper, pose_to_eef  # noqa: E402
from deploy.utils.va_action_bridge import VAActionBridge  # noqa: E402
from deploy.utils.va_observation import build_va_frame  # noqa: E402

DEFAULT_PROFILE_CONFIG = CLIENT_ROOT / 'launch_profiles.yaml'
DEFAULT_PROFILE_NAME = 'lift2_va_default'
DEFAULT_LANGUAGE_INSTRUCTION = 'perform task'

PRESET_TASK_INSTRUCTIONS = {
    'tube': 'Transfer the test tube from the right rack to the left rack.',
    'towel': 'Fold the towel.',
    'wrench': 'Open the toolbox, check the items inside one by one, and find the wrench.',
    'power_strip': (
        'Move the power strip with the left arm, and press the button of the '
        'power strip with the right arm.'
    ),
    'drum': 'Pick up two small drumsticks and hit the small drum.',
    'dice': (
        'Roll the dice and move the small stand the specified number of squares '
        'based on the number rolled.'
    ),
    'stack': 'Stack the building blocks one by one with the larger ones at the bottom.',
    'size': (
        'Pick up the four randomly placed cylinders and insert each one into the '
        'matching hole according to its size.'
    ),
    'color': (
        'Pick up each colored cylinder placed in front of the base and insert it '
        'into the empty groove at the matching color position on the 4-by-4 board.'
    ),
}

PROFILE_REQUIRED_KEYS = ('host', 'port', 'publish_rate')
PROFILE_OPTIONAL_KEYS = (
    'camera_mode',
    'bridge_mode',
    'first_chunk_start_idx',
    'keyframe_divisor',
    'gripper_raw_max',
    'executor_rate_hz',
    'max_publish_step',
    'dry_run_mock_chunks',
    'connect_retry_seconds',
    'connect_timeout_seconds',
    'metadata_timeout_seconds',
    'server_wait_timeout_seconds',
    'auto_init',
    'init_duration',
    'wait_after_init',
    'left_init_pose',
    'right_init_pose',
)
PROFILE_VALUE_CASTERS = {
    'host': str,
    'port': int,
    'publish_rate': int,
    'camera_mode': str,
    'bridge_mode': str,
    'first_chunk_start_idx': int,
    'keyframe_divisor': int,
    'gripper_raw_max': float,
    'executor_rate_hz': float,
    'max_publish_step': int,
    'dry_run_mock_chunks': int,
    'connect_retry_seconds': float,
    'connect_timeout_seconds': float,
    'metadata_timeout_seconds': float,
    'server_wait_timeout_seconds': float,
    'auto_init': bool,
    'init_duration': float,
    'wait_after_init': bool,
    'left_init_pose': list,
    'right_init_pose': list,
}

DEFAULT_ROS_TOPICS = {
    'img_front_topic': '/camera_h/color/image_raw',
    'img_left_topic': '/camera_l/color/image_raw',
    'img_right_topic': '/camera_r/color/image_raw',
    'img_front_depth_topic': '/camera_h/aligned_depth_to_color/image_raw',
    'img_left_depth_topic': '/camera_l/aligned_depth_to_color/image_raw',
    'img_right_depth_topic': '/camera_r/aligned_depth_to_color/image_raw',
    'arm_left_pose_topic': '/arm_left/arm_status_ee',
    'arm_right_pose_topic': '/arm_right/arm_status_ee',
    'arm_left_cmd_topic': '/arm_left_cmd',
    'arm_right_cmd_topic': '/arm_right_cmd',
}


def resolve_language_instruction(args: argparse.Namespace) -> str:
    if args.language_instruction is not None:
        return args.language_instruction
    if args.task:
        return PRESET_TASK_INSTRUCTIONS[args.task]
    return DEFAULT_LANGUAGE_INSTRUCTION


def load_launch_profile(config_path: Path, profile_name: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            f'PyYAML is required to read {config_path}: {exc}'
        ) from exc

    with open(config_path, 'r', encoding='utf-8') as file_handle:
        data = yaml.safe_load(file_handle) or {}

    profiles = data.get('profiles')
    if not isinstance(profiles, dict):
        raise ValueError(f"Invalid config: missing 'profiles' in {config_path}")

    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        available = ', '.join(sorted(profiles)) or '<none>'
        raise ValueError(
            f"Unknown profile '{profile_name}'. Available: {available}"
        )

    missing = [key for key in PROFILE_REQUIRED_KEYS if key not in profile]
    if missing:
        raise ValueError(
            f"Profile '{profile_name}' missing keys: {', '.join(missing)}"
        )
    return dict(profile)


def apply_launch_profile(args: argparse.Namespace) -> None:
    if not getattr(args, 'profile', None):
        return
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    profile = load_launch_profile(config_path, args.profile)
    args.profile_config_path = str(config_path)

    for key in PROFILE_REQUIRED_KEYS + PROFILE_OPTIONAL_KEYS:
        if key in profile and getattr(args, key, None) is None:
            caster = PROFILE_VALUE_CASTERS.get(key, str)
            setattr(args, key, caster(profile[key]))


def finalize_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    apply_launch_profile(args)

    if args.host is None:
        args.host = '127.0.0.1'
    if args.port is None:
        args.port = 7777
    if args.publish_rate is None:
        args.publish_rate = 60
    if args.camera_mode is None:
        args.camera_mode = 'tshape'
    if args.bridge_mode is None:
        args.bridge_mode = 'segment_relative_va'
    if args.first_chunk_start_idx is None:
        args.first_chunk_start_idx = 1
    if args.keyframe_divisor is None:
        args.keyframe_divisor = 4
    if args.gripper_raw_max is None:
        args.gripper_raw_max = 5.0
    if args.executor_rate_hz is None:
        args.executor_rate_hz = float(args.publish_rate)
    if args.max_publish_step is None:
        args.max_publish_step = 0
    if args.dry_run_mock_chunks is None:
        args.dry_run_mock_chunks = 2
    if args.connect_retry_seconds is None:
        args.connect_retry_seconds = 5.0
    if args.connect_timeout_seconds is None:
        args.connect_timeout_seconds = 10.0
    if args.metadata_timeout_seconds is None:
        args.metadata_timeout_seconds = 10.0
    if args.server_wait_timeout_seconds is None:
        # Zero means keep retrying until Ctrl+C. Each individual connection
        # and metadata wait still has a finite timeout.
        args.server_wait_timeout_seconds = 0.0
    if args.auto_init is None:
        args.auto_init = True
    if args.init_duration is None:
        args.init_duration = 3.0
    if args.wait_after_init is None:
        args.wait_after_init = False
    if args.left_init_pose is None:
        args.left_init_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    if args.right_init_pose is None:
        args.right_init_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    for pose_name in ('left_init_pose', 'right_init_pose'):
        pose = np.asarray(getattr(args, pose_name), dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(f'{pose_name} must contain exactly 7 values, got {pose.shape}')
        setattr(args, pose_name, pose.tolist())

    args.language_instruction = resolve_language_instruction(args)

    for topic_key, topic_value in DEFAULT_ROS_TOPICS.items():
        if getattr(args, topic_key, None) is None:
            setattr(args, topic_key, topic_value)

    if args.use_depth_image is None:
        args.use_depth_image = False

    return args


def wait_for_synced_frame(
    ros_operator,
    publish_rate: int,
    log_fn=print,
    timeout_seconds: float = 30.0,
):
    """Wait until cameras + arms are ready, then return one synced frame.

    Uses ``get_frame()`` (pop) only for the **initial** observation so the
    first packet is time-aligned. Keyframes during execution use
    ``get_latest_images()`` (peek) instead.
    """
    period = 1.0 / float(publish_rate)
    warned = False
    deadline = time.time() + float(timeout_seconds)
    while True:
        result = ros_operator.get_frame()
        if result:
            return result
        if time.time() >= deadline:
            raise TimeoutError(
                f'Timed out after {timeout_seconds:.1f}s waiting for '
                'synchronized camera + arm state'
            )
        if not warned:
            log_fn('Waiting for synchronized camera + arm state...')
            warned = True
        time.sleep(period)


def move_to_init_pose(
    ros_operator,
    left_init_pose,
    right_init_pose,
    duration_seconds: float,
    publish_rate: float,
    gripper_raw_max: float,
    log_fn=print,
    sleep_fn=time.sleep,
) -> None:
    """Wait for dual-arm state, then smoothly move both arms to reset pose."""
    if duration_seconds <= 0:
        raise ValueError('init_duration must be positive')
    if publish_rate <= 0:
        raise ValueError('publish_rate must be positive')

    left_target = np.asarray(left_init_pose, dtype=np.float64).reshape(7)
    right_target = np.asarray(right_init_pose, dtype=np.float64).reshape(7)
    wait_timeout_seconds = max(float(duration_seconds), 5.0)
    wait_deadline = time.time() + wait_timeout_seconds
    control_period_seconds = 1.0 / float(publish_rate)
    logged_wait = False

    log_fn('[init] Waiting for arm state before reset...')
    while True:
        latest_arm_poses = ros_operator.get_latest_arm_poses()
        if latest_arm_poses:
            break
        if time.time() >= wait_deadline:
            raise TimeoutError('Timed out waiting for arm state before reset')
        if not logged_wait:
            log_fn('[init] Arm state is not ready; waiting...')
            logged_wait = True
        sleep_fn(control_period_seconds)

    arm_left_pose, arm_right_pose = latest_arm_poses
    current_eef14 = pose_to_eef(
        arm_left_pose,
        arm_right_pose,
        gripper_max=gripper_raw_max,
    )
    left_current = current_eef14[:7]
    right_current = current_eef14[7:14]

    log_fn(
        f'[init] Resetting arms: L {left_current[:3]} -> {left_target[:3]}, '
        f'R {right_current[:3]} -> {right_target[:3]}'
    )
    total_steps = max(1, int(float(duration_seconds) * float(publish_rate)))
    progress_log_interval = max(1, int(float(publish_rate)))

    for step_index in range(total_steps + 1):
        interpolation_ratio = step_index / total_steps
        left_command = (
            left_current * (1.0 - interpolation_ratio)
            + left_target * interpolation_ratio
        )
        right_command = (
            right_current * (1.0 - interpolation_ratio)
            + right_target * interpolation_ratio
        )

        left_command_raw = left_command.copy()
        right_command_raw = right_command.copy()
        left_command_raw[6] = denormalize_gripper(
            left_command[6], gripper_max=gripper_raw_max
        )
        right_command_raw[6] = denormalize_gripper(
            right_command[6], gripper_max=gripper_raw_max
        )
        ros_operator.eef_arm_publish(
            left_command_raw.tolist(),
            right_command_raw.tolist(),
        )

        if step_index % progress_log_interval == 0 or step_index == total_steps:
            progress_percent = int(round(interpolation_ratio * 100.0))
            log_fn(
                f'[init] Reset progress: {progress_percent}% '
                f'({step_index}/{total_steps})'
            )
        if step_index < total_steps:
            sleep_fn(control_period_seconds)

    log_fn('[init] Reached reset pose')


def create_websocket_policy(args: argparse.Namespace) -> WebsocketVAPolicy:
    """Create a diagnostic-friendly WebSocket client from runtime arguments."""
    max_wait_seconds = (
        None
        if args.server_wait_timeout_seconds <= 0
        else args.server_wait_timeout_seconds
    )
    return WebsocketVAPolicy(
        host=args.host,
        port=args.port,
        connect_retry_seconds=args.connect_retry_seconds,
        connect_timeout_seconds=args.connect_timeout_seconds,
        metadata_timeout_seconds=args.metadata_timeout_seconds,
        max_wait_seconds=max_wait_seconds,
    )


def publish_absolute_eef14(ros_operator, absolute_eef14_raw: np.ndarray) -> None:
    action = np.asarray(absolute_eef14_raw, dtype=np.float64).reshape(14)
    ros_operator.eef_arm_publish(action[:7].tolist(), action[7:14].tolist())


def sample_keyframe_va(
    ros_operator,
    camera_mode: str,
    task: Optional[str],
    step_count: int,
) -> Dict[str, Any]:
    """Sample one VA keyframe by peeking latest RGB images (no queue pop).

    Raises if cameras are empty so the root cause is not hidden by silent
    fallbacks or empty ``compute_kv_cache`` payloads.
    """
    latest = ros_operator.get_latest_images()
    if not latest:
        raise RuntimeError(
            f'Keyframe sample failed at step={step_count}: camera queues empty. '
            'Check RealSense topics and RosOperator subscriptions.'
        )
    img_front, img_left, img_right = latest
    return build_va_frame(
        img_front,
        img_left,
        img_right,
        camera_mode=camera_mode,
        task=task,
    )


def run_episode(
    args: argparse.Namespace,
    policy: WebsocketVAPolicy,
    ros_operator,
    bridge: VAActionBridge,
    log_fn=print,
) -> None:
    if args.auto_init:
        if args.dry_run:
            log_fn('[init] Skipping reset motion because --dry_run is enabled')
        else:
            move_to_init_pose(
                ros_operator=ros_operator,
                left_init_pose=args.left_init_pose,
                right_init_pose=args.right_init_pose,
                duration_seconds=args.init_duration,
                publish_rate=args.publish_rate,
                gripper_raw_max=args.gripper_raw_max,
                log_fn=log_fn,
            )
            if args.wait_after_init:
                input('Reset complete. Press Enter to start evaluation...')

    prompt = args.language_instruction
    log_fn(f'[VA] reset prompt={prompt!r}')
    policy.reset(prompt=prompt)

    (
        img_front,
        img_left,
        img_right,
        _img_front_depth,
        _img_left_depth,
        _img_right_depth,
        arm_left_pose,
        arm_right_pose,
    ) = wait_for_synced_frame(ros_operator, args.publish_rate, log_fn=log_fn)

    init_eef14 = pose_to_eef(
        arm_left_pose,
        arm_right_pose,
        gripper_max=args.gripper_raw_max,
    )
    bridge.reset(init_eef14)
    log_fn(
        f'[VA] cached init EEF xyz L={init_eef14[:3]} R={init_eef14[7:10]} '
        f'grip=({init_eef14[6]:.3f},{init_eef14[13]:.3f})'
    )

    # Images from RosOperator are RGB (openpi real-robot convention).
    first_obs = build_va_frame(
        img_front,
        img_left,
        img_right,
        camera_mode=args.camera_mode,
        task=prompt,
    )
    # Same as robotwin simulation: reuse first_obs for every infer call;
    # executed-world observations enter the model only via compute_kv_cache.
    policy_obs = first_obs

    period = 1.0 / float(args.publish_rate)
    step_count = 0
    chunk_count = 0
    run_infinite = args.max_publish_step <= 0

    try:
        import rospy as _rospy_for_shutdown
    except ImportError:
        _rospy_for_shutdown = None

    while run_infinite or step_count < args.max_publish_step:
        if _rospy_for_shutdown is not None and _rospy_for_shutdown.is_shutdown():
            break

        t_infer0 = time.perf_counter()
        result = policy.infer_action(policy_obs)
        action = np.asarray(result['action'])
        infer_ms = (time.perf_counter() - t_infer0) * 1000.0
        log_fn(
            f'[VA] chunk={chunk_count} action.shape={action.shape} '
            f'infer_ms={infer_ms:.1f}'
        )

        micro_steps = bridge.expand(action)
        key_frames: List[Dict[str, Any]] = []

        for micro_step in micro_steps:
            if (not run_infinite) and step_count >= args.max_publish_step:
                break

            # Rate-limit absolute EEF publishes to publish_rate (default 60 Hz).
            # Without this sleep the client would spam PosCmd as fast as the
            # CPU allows, ignoring the 60 Hz data / control contract.
            t_pub0 = time.perf_counter()
            if args.dry_run:
                if step_count % max(1, int(args.publish_rate)) == 0:
                    raw = micro_step.absolute_eef14_raw
                    log_fn(
                        f'[dry-run step {step_count}] '
                        f'L_xyz={raw[:3]} R_xyz={raw[7:10]} '
                        f'grip=({raw[6]:.2f},{raw[13]:.2f})'
                    )
            else:
                publish_absolute_eef14(ros_operator, micro_step.absolute_eef14_raw)

            if micro_step.is_keyframe:
                key_frames.append(
                    sample_keyframe_va(
                        ros_operator,
                        camera_mode=args.camera_mode,
                        task=prompt,
                        step_count=step_count,
                    )
                )

            step_count += 1
            elapsed = time.perf_counter() - t_pub0
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

        if not key_frames:
            # Fail hard: empty list would make VA_Server._encode_obs return None
            # and crash deeper with a less actionable stack trace.
            raise RuntimeError(
                f'compute_kv_cache refused: key_frames is empty after chunk '
                f'{chunk_count} (executed_steps={len(micro_steps)}, '
                f'action.shape={action.shape}). '
                'No keyframe was sampled; check camera topics / keyframe schedule.'
            )

        log_fn(
            f'[VA] compute_kv_cache keyframes={len(key_frames)} '
            f'state_shape={action.shape}'
        )
        policy.compute_kv_cache(key_frames, state=action)
        chunk_count += 1

        if args.verbose:
            log_fn(f'[VA] completed chunks={chunk_count} steps={step_count}')


def build_mock_frame(
    camera_mode: str,
    task: str,
    frame_index: int,
) -> Dict[str, Any]:
    """Build deterministic RGB camera data with the real observation schema."""
    frame_value = int(frame_index) % 256

    head_rgb = np.empty((480, 640, 3), dtype=np.uint8)
    head_rgb[...] = (
        frame_value,
        (frame_value + 31) % 256,
        (frame_value + 63) % 256,
    )

    left_wrist_rgb = np.empty((240, 320, 3), dtype=np.uint8)
    left_wrist_rgb[...] = (
        (frame_value + 97) % 256,
        (frame_value + 127) % 256,
        (frame_value + 159) % 256,
    )

    right_wrist_rgb = np.empty((240, 320, 3), dtype=np.uint8)
    right_wrist_rgb[...] = (
        (frame_value + 193) % 256,
        (frame_value + 211) % 256,
        (frame_value + 239) % 256,
    )

    return build_va_frame(
        head_rgb,
        left_wrist_rgb,
        right_wrist_rgb,
        camera_mode=camera_mode,
        task=task,
    )


def run_mock_episode(
    args: argparse.Namespace,
    policy: WebsocketVAPolicy,
    log_fn=print,
) -> List[Dict[str, Any]]:
    """Run the complete client loop with deterministic fake robot data.

    The WebSocket requests and server responses are real. The initial camera
    frame, simulated action execution, and sparse keyframes are synthetic, so
    this path exercises protocol state transitions without ROS or hardware.
    """
    mock_chunk_count = int(args.dry_run_mock_chunks)
    if mock_chunk_count < 1:
        raise ValueError(
            f'dry_run_mock_chunks must be at least 1, got {mock_chunk_count}'
        )

    prompt = args.language_instruction
    log_fn(f'[dry-run-mock] reset prompt={prompt!r}')
    policy.reset(prompt=prompt)

    initial_frame = build_mock_frame(
        args.camera_mode,
        prompt,
        frame_index=0,
    )
    bridge = VAActionBridge(
        first_chunk_start_idx=args.first_chunk_start_idx,
        keyframe_divisor=args.keyframe_divisor,
        gripper_raw_max=args.gripper_raw_max,
        bridge_mode=args.bridge_mode,
    )
    init_eef14 = np.array(
        [0, 0, 0.2, 0, 0, 0, 1.0, 0, 0, 0.2, 0, 0, 0, 1.0],
        dtype=np.float64,
    )
    bridge.reset(init_eef14)

    summaries: List[Dict[str, Any]] = []
    fake_execution_step = 0

    for chunk_index in range(mock_chunk_count):
        infer_started_at = time.perf_counter()
        result = policy.infer_action(initial_frame)
        infer_ms = (time.perf_counter() - infer_started_at) * 1000.0
        action = np.asarray(result['action'])
        log_fn(
            f'[dry-run-mock] chunk={chunk_index} '
            f'infer action.shape={action.shape} dtype={action.dtype} '
            f'infer_ms={infer_ms:.1f}'
        )

        micro_steps = bridge.expand(action)
        key_frames: List[Dict[str, Any]] = []
        last_absolute_eef14_raw: Optional[np.ndarray] = None

        for micro_step in micro_steps:
            fake_execution_step += 1
            last_absolute_eef14_raw = micro_step.absolute_eef14_raw

            if micro_step.is_keyframe:
                key_frames.append(
                    build_mock_frame(
                        args.camera_mode,
                        prompt,
                        frame_index=fake_execution_step,
                    )
                )

        if not key_frames:
            raise RuntimeError(
                f'[dry-run-mock] no keyframes generated for chunk={chunk_index}; '
                f'action.shape={action.shape}'
            )

        kv_started_at = time.perf_counter()
        kv_response = policy.compute_kv_cache(key_frames, state=action)
        kv_ms = (time.perf_counter() - kv_started_at) * 1000.0
        if not isinstance(kv_response, dict):
            raise RuntimeError(
                '[dry-run-mock] compute_kv_cache returned '
                f'{type(kv_response).__name__}, expected dict'
            )

        summary = {
            'chunk_index': chunk_index,
            'action_shape': tuple(action.shape),
            'executed_steps': len(micro_steps),
            'keyframe_count': len(key_frames),
            'kv_response_keys': tuple(sorted(kv_response.keys())),
        }
        summaries.append(summary)
        log_fn(
            f'[dry-run-mock] chunk={chunk_index} '
            f'fake_execute_steps={len(micro_steps)} '
            f'keyframes={len(key_frames)} '
            f'compute_kv_cache_keys={list(summary["kv_response_keys"])} '
            f'kv_ms={kv_ms:.1f}'
        )
        if last_absolute_eef14_raw is not None and args.verbose:
            log_fn(
                f'[dry-run-mock] chunk={chunk_index} '
                f'last_fake_raw_eef={last_absolute_eef14_raw}'
            )

    return summaries


def run_dry_run_mock(args: argparse.Namespace) -> None:
    """No ROS: execute the complete protocol loop with fake robot data."""
    print(f'[dry-run-mock] connecting to {args.host}:{args.port}')
    policy = create_websocket_policy(args)
    try:
        print(f'[dry-run-mock] metadata={policy.get_server_metadata()}')
        summaries = run_mock_episode(args, policy, log_fn=print)
        total_fake_execution_steps = sum(
            item['executed_steps'] for item in summaries
        )
        total_keyframes = sum(item['keyframe_count'] for item in summaries)
        print(
            f'[dry-run-mock] PASS chunks={len(summaries)} '
            f'fake_execution_steps={total_fake_execution_steps} '
            f'keyframes={total_keyframes}'
        )
    finally:
        policy.close()


def run_server_probe(args: argparse.Namespace) -> None:
    """Verify the WebSocket handshake and metadata without model inference."""
    print(f'[server-probe] connecting to {args.host}:{args.port}', flush=True)
    policy = create_websocket_policy(args)
    try:
        print(
            f'[server-probe] WebSocket handshake succeeded; '
            f'metadata={policy.get_server_metadata()}',
            flush=True,
        )
        print('[server-probe] PASS', flush=True)
    finally:
        policy.close()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='LIFT2 real-robot client for LingBot-VA / 4DWAM'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(DEFAULT_PROFILE_CONFIG),
        help='Path to client launch_profiles.yaml',
    )
    parser.add_argument(
        '--profile',
        type=str,
        default=DEFAULT_PROFILE_NAME,
        help='Profile name inside launch_profiles.yaml',
    )
    parser.add_argument('--host', type=str, default=None)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--publish_rate', type=int, default=None)
    parser.add_argument('--camera_mode', type=str, default=None)
    parser.add_argument('--bridge_mode', type=str, default=None)
    parser.add_argument('--first_chunk_start_idx', type=int, default=None)
    parser.add_argument('--keyframe_divisor', type=int, default=None)
    parser.add_argument('--gripper_raw_max', type=float, default=None)
    parser.add_argument('--executor_rate_hz', type=float, default=None)
    parser.add_argument('--max_publish_step', type=int, default=None)
    parser.add_argument('--connect_retry_seconds', type=float, default=None)
    parser.add_argument('--connect_timeout_seconds', type=float, default=None)
    parser.add_argument('--metadata_timeout_seconds', type=float, default=None)
    parser.add_argument(
        '--server_wait_timeout_seconds',
        type=float,
        default=None,
        help='Total server wait timeout; 0 means wait until Ctrl+C (default: 0).',
    )
    parser.add_argument(
        '--auto_init',
        action='store_true',
        default=None,
        help='Move both arms to the configured reset pose before evaluation.',
    )
    parser.add_argument(
        '--no_auto_init',
        action='store_false',
        dest='auto_init',
        help='Skip the startup reset motion.',
    )
    parser.add_argument(
        '--init_duration',
        type=float,
        default=None,
        help='Seconds used to interpolate from current state to reset pose.',
    )
    parser.add_argument(
        '--wait_after_init',
        action='store_true',
        default=None,
        help='Wait for Enter after reset completes and before evaluation starts.',
    )
    parser.add_argument(
        '--left_init_pose',
        type=float,
        nargs=7,
        default=None,
        metavar=('X', 'Y', 'Z', 'ROLL', 'PITCH', 'YAW', 'GRIPPER'),
        help='Left reset EEF pose; gripper is normalized to [0, 1].',
    )
    parser.add_argument(
        '--right_init_pose',
        type=float,
        nargs=7,
        default=None,
        metavar=('X', 'Y', 'Z', 'ROLL', 'PITCH', 'YAW', 'GRIPPER'),
        help='Right reset EEF pose; gripper is normalized to [0, 1].',
    )
    parser.add_argument(
        '--dry_run_mock_chunks',
        type=int,
        default=None,
        help=(
            'Number of fake action chunks to execute in --dry_run_mock '
            '(default: 2).'
        ),
    )
    parser.add_argument('--task', type=str, choices=sorted(PRESET_TASK_INSTRUCTIONS), default=None)
    parser.add_argument('--language_instruction', type=str, default=None)
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Connect to ROS if available but do not publish arm commands '
        '(still samples keyframes).',
    )
    parser.add_argument(
        '--dry_run_mock',
        action='store_true',
        help='No ROS: fake images against a live VA server (GPU dry-run).',
    )
    parser.add_argument(
        '--probe_server',
        action='store_true',
        help=(
            'Only test the WebSocket handshake and metadata; do not use ROS '
            'or run model inference.'
        ),
    )
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use_depth_image', action='store_true', default=None)

    for topic_key in DEFAULT_ROS_TOPICS:
        parser.add_argument(f'--{topic_key}', type=str, default=None)

    args = parser.parse_args(argv)
    return finalize_runtime_args(args)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    print('=' * 50)
    print('LIFT2 VA real-robot client')
    print('=' * 50)
    if getattr(args, 'profile', None):
        print(f"Profile: {args.profile} ({getattr(args, 'profile_config_path', args.config)})")
    print(f'Policy server: {args.host}:{args.port}')
    print(f'Publish rate: {args.publish_rate} Hz')
    print(f'Camera mode: {args.camera_mode}')
    print(f'Bridge mode: {args.bridge_mode}')
    print(f'Language: {args.language_instruction}')
    print(
        f'Auto reset: {args.auto_init} '
        f'(duration={args.init_duration:.1f}s, wait_after={args.wait_after_init})'
    )
    print(
        f'WebSocket timeouts: connect={args.connect_timeout_seconds:.1f}s '
        f'metadata={args.metadata_timeout_seconds:.1f}s '
        f'total={args.server_wait_timeout_seconds:.1f}s '
        '(0 means unlimited)'
    )
    print(f'dry_run={args.dry_run} dry_run_mock={args.dry_run_mock}')
    print('=' * 50)

    try:
        if args.probe_server:
            run_server_probe(args)
            return 0

        if args.dry_run_mock:
            # WebsocketVAPolicy performs the actual WebSocket handshake and
            # retries until the server is ready. Do not preflight with a raw
            # TCP connect: it would make the WebSocket server log an invalid
            # HTTP request because no HTTP Upgrade request was sent.
            run_dry_run_mock(args)
            return 0

        try:
            import rospy
        except ImportError as exc:
            print(
                'rospy is required for real-robot mode. '
                f'Use --dry_run_mock for server-only tests. ({exc})'
            )
            return 1

        print('[client] importing RosOperator...', flush=True)
        from deploy.utils.rosoperator import RosOperator

        print('[client] initializing ROS node...', flush=True)
        rospy.init_node('lift2_va_client', anonymous=True)
        print('[client] creating RosOperator...', flush=True)
        ros_operator = RosOperator(args)

        bridge = VAActionBridge(
            first_chunk_start_idx=args.first_chunk_start_idx,
            keyframe_divisor=args.keyframe_divisor,
            gripper_raw_max=args.gripper_raw_max,
            bridge_mode=args.bridge_mode,
        )

        print(
            f'[client] opening WebSocket to {args.host}:{args.port}...',
            flush=True,
        )
        policy = create_websocket_policy(args)
        rospy.loginfo(f'Server metadata: {policy.get_server_metadata()}')

        try:
            run_episode(args, policy, ros_operator, bridge, log_fn=rospy.loginfo)
        finally:
            policy.close()
            rospy.loginfo('Client stopped')
    except KeyboardInterrupt:
        print('\nClient interrupted by user.')
        return 130
    except TimeoutError as error:
        print(f'Client startup failed: {error}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
