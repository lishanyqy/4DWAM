#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference-only high-rate EEF action executor for OpenPI LIFT2.

The policy/control loop can keep producing 14D absolute EEF targets at a
lower rate while this executor publishes smooth intermediate commands at a
higher ROS command rate.
"""

import collections
import queue
import threading
import time

import numpy as np
import rospy


class EEFInterpolatingExecutor:
    """
    Background EEF command executor with minimum-jerk interpolation.

    Action format:
        [left_x, left_y, left_z, left_roll, left_pitch, left_yaw, left_gripper,
         right_x, right_y, right_z, right_roll, right_pitch, right_yaw, right_gripper]
    """

    RPY_INDICES = (3, 4, 5, 10, 11, 12)
    GRIPPER_INDICES = (6, 13)

    def __init__(
        self,
        ros_operator,
        policy_rate_hz=30.0,
        executor_rate_hz=90.0,
        interpolation='minimum_jerk',
        gripper_mode='passthrough',
        max_queue_size=60,
        log_prefix='[EEFExecutor]',
    ):
        self.ros_operator = ros_operator
        self.policy_rate_hz = float(policy_rate_hz)
        self.executor_rate_hz = float(executor_rate_hz)
        self.interpolation = interpolation
        self.gripper_mode = gripper_mode
        self.max_queue_size = int(max_queue_size)
        self.log_prefix = log_prefix

        if self.policy_rate_hz <= 0:
            raise ValueError(f'policy_rate_hz must be positive, got {self.policy_rate_hz}')
        if self.executor_rate_hz <= 0:
            raise ValueError(f'executor_rate_hz must be positive, got {self.executor_rate_hz}')
        if self.max_queue_size <= 0:
            raise ValueError(f'max_queue_size must be positive, got {self.max_queue_size}')
        if self.interpolation not in ('minimum_jerk', 'linear'):
            raise ValueError(f'Unknown interpolation: {self.interpolation}')
        if self.gripper_mode not in ('passthrough', 'interp'):
            raise ValueError(f'Unknown gripper_mode: {self.gripper_mode}')

        ratio = self.executor_rate_hz / self.policy_rate_hz
        self.substeps = max(1, int(round(ratio)))
        if abs(ratio - self.substeps) > 1e-3:
            rospy.logwarn(
                f'{self.log_prefix} executor_rate_hz should preferably be an integer '
                f'multiple of policy_rate_hz. Got ratio={ratio:.3f}; '
                f'using substeps={self.substeps}.'
            )

        self.action_queue = queue.Queue(maxsize=self.max_queue_size)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

        self.last_cmd = None
        self.segment_start = None
        self.segment_target = None
        self.segment_step = 0
        self.loop_active = False

    def start(self, initial_action=None):
        if self.is_running():
            return

        self.reset(initial_action)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        rospy.loginfo(
            f'{self.log_prefix} Started: '
            f'policy_rate={self.policy_rate_hz:.1f}Hz, '
            f'executor_rate={self.executor_rate_hz:.1f}Hz, '
            f'substeps={self.substeps}, '
            f'interpolation={self.interpolation}, '
            f'gripper_mode={self.gripper_mode}'
        )

    def stop(self, drain=False, drain_timeout=None):
        if self.thread is None:
            return True

        drained = True
        if drain and self.thread.is_alive():
            drained = self.drain(timeout=drain_timeout)

        self.stop_event.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            rospy.logerr(
                f'{self.log_prefix} Stop requested but executor thread did not exit within 2.0s. '
                'It may still publish commands; skipping any direct homing publish is recommended.'
            )
            return False

        self.thread = None
        rospy.loginfo(f'{self.log_prefix} Stopped')
        return drained

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def reset(self, initial_action=None):
        with self.lock:
            self._clear_queue_locked()
            self.last_cmd = None if initial_action is None else self._validate_action(initial_action).copy()
            self.segment_start = None
            self.segment_target = None
            self.segment_step = 0
            self.loop_active = False

    def enqueue(self, action):
        action = self._validate_action(action)

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            try:
                self.action_queue.put(action.copy(), timeout=0.1)
                return
            except queue.Full:
                rospy.logwarn_throttle(
                    1.0,
                    f'{self.log_prefix} Action queue full ({self.max_queue_size}); '
                    'waiting for executor to catch up without dropping actions.',
                )

    def drain(self, timeout=None):
        if timeout is None:
            timeout = self._default_drain_timeout()

        deadline = time.time() + float(timeout)
        sleep_s = min(max(1.0 / self.executor_rate_hz, 0.001), 0.05)

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            with self.lock:
                segment_done = self.segment_target is None or self.segment_step >= self.substeps
                drained = self.action_queue.empty() and segment_done and not self.loop_active

            if drained:
                return True

            if time.time() >= deadline:
                rospy.logwarn(
                    f'{self.log_prefix} Timed out waiting for executor drain after {timeout:.2f}s. '
                    'Stopping with pending commands may skip the last endpoint.'
                )
                return False

            time.sleep(sleep_s)

        return False

    def _default_drain_timeout(self):
        queue_s = float(self.max_queue_size) / max(self.policy_rate_hz, 1e-6)
        segment_s = float(self.substeps + 1) / max(self.executor_rate_hz, 1e-6)
        return max(2.0, queue_s + segment_s + 1.0)

    def _clear_queue_locked(self):
        while True:
            try:
                self.action_queue.get_nowait()
            except queue.Empty:
                break

    def _validate_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (14,):
            raise ValueError(f'Expected action shape (14,), got {action.shape}')
        if not np.all(np.isfinite(action)):
            raise ValueError(f'Action contains NaN/Inf: {action}')
        return action

    def _run_loop(self):
        rate = rospy.Rate(self.executor_rate_hz)
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            try:
                with self.lock:
                    self.loop_active = True
                cmd = self._next_command()
                if cmd is not None:
                    self._publish(cmd)
            except Exception as exc:
                rospy.logerr(f'{self.log_prefix} Error in executor loop: {exc}')
            finally:
                with self.lock:
                    self.loop_active = False
            rate.sleep()

    def _next_command(self):
        with self.lock:
            if self.last_cmd is None:
                target = self._try_get_target()
                if target is None:
                    return None
                self.last_cmd = target.copy()
                return self.last_cmd.copy()

            if self.segment_target is None or self.segment_step >= self.substeps:
                target = self._try_get_target()
                if target is None:
                    return self.last_cmd.copy()

                self.segment_start = self.last_cmd.copy()
                self.segment_target = self._unwrap_target(self.segment_start, target)
                self.segment_step = 0

            self.segment_step += 1
            u = float(self.segment_step) / float(self.substeps)
            s = self._interp_scalar(u)
            cmd = self.segment_start + s * (self.segment_target - self.segment_start)

            if self.gripper_mode == 'passthrough':
                for index in self.GRIPPER_INDICES:
                    cmd[index] = self.segment_target[index]

            if self.segment_step >= self.substeps:
                cmd = self.segment_target.copy()

            self.last_cmd = cmd.copy()
            return cmd

    def _try_get_target(self):
        qsize = self.action_queue.qsize()
        if qsize > self.max_queue_size * 0.5:
            rospy.logwarn_throttle(
                1.0,
                f'{self.log_prefix} Queue backlog: {qsize}/{self.max_queue_size}',
            )

        try:
            return self.action_queue.get_nowait()
        except queue.Empty:
            return None

    def _interp_scalar(self, u):
        u = min(max(float(u), 0.0), 1.0)
        if self.interpolation == 'linear':
            return u
        if self.interpolation == 'minimum_jerk':
            return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        raise ValueError(f'Unknown interpolation: {self.interpolation}')

    def _unwrap_target(self, start, target):
        target = target.copy()
        for index in self.RPY_INDICES:
            target[index] = self._shortest_angle_target(start[index], target[index])
        return target

    @staticmethod
    def _shortest_angle_target(start_angle, target_angle):
        delta = (target_angle - start_angle + np.pi) % (2.0 * np.pi) - np.pi
        return start_angle + delta

    def _publish(self, cmd):
        self.ros_operator.eef_arm_publish(cmd[:7], cmd[7:14])


class EEFTrajectoryExecutor:
    """
    Background EEF trajectory player that preserves policy waypoint semantics.

    Unlike EEFInterpolatingExecutor, this executor never treats the previously
    published command as the semantic start of the next segment. Segment endpoints
    are adjacent policy waypoints:

        waypoint_i -> waypoint_i+1

    If the executor thread stalls, playback stretches in wall time because segment
    progress advances by executor ticks, not by elapsed wall-clock time.
    """

    RPY_INDICES = EEFInterpolatingExecutor.RPY_INDICES
    GRIPPER_INDICES = EEFInterpolatingExecutor.GRIPPER_INDICES

    def __init__(
        self,
        ros_operator,
        policy_rate_hz=30.0,
        executor_rate_hz=90.0,
        interpolation='linear',
        gripper_mode='passthrough',
        max_queue_size=60,
        log_prefix='[EEFTrajectoryExecutor]',
    ):
        self.ros_operator = ros_operator
        self.policy_rate_hz = float(policy_rate_hz)
        self.executor_rate_hz = float(executor_rate_hz)
        self.interpolation = interpolation
        self.gripper_mode = gripper_mode
        self.max_queue_size = int(max_queue_size)
        self.log_prefix = log_prefix

        if self.policy_rate_hz <= 0:
            raise ValueError(f'policy_rate_hz must be positive, got {self.policy_rate_hz}')
        if self.executor_rate_hz <= 0:
            raise ValueError(f'executor_rate_hz must be positive, got {self.executor_rate_hz}')
        if self.max_queue_size <= 0:
            raise ValueError(f'max_queue_size must be positive, got {self.max_queue_size}')
        if self.interpolation not in ('minimum_jerk', 'linear'):
            raise ValueError(f'Unknown interpolation: {self.interpolation}')
        if self.gripper_mode not in ('passthrough', 'interp'):
            raise ValueError(f'Unknown gripper_mode: {self.gripper_mode}')

        ratio = self.executor_rate_hz / self.policy_rate_hz
        self.substeps = max(1, int(round(ratio)))
        if abs(ratio - self.substeps) > 1e-3:
            rospy.logwarn(
                f'{self.log_prefix} executor_rate_hz should preferably be an integer '
                f'multiple of policy_rate_hz. Got ratio={ratio:.3f}; '
                f'using substeps={self.substeps}.'
            )

        self.waypoint_buffer = collections.deque()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

        self.anchor_waypoint = None
        self.segment_start = None
        self.segment_target = None
        self.segment_step = 0
        self.loop_active = False

    def start(self, initial_action=None):
        if self.is_running():
            return

        self.reset(initial_action)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        rospy.loginfo(
            f'{self.log_prefix} Started: '
            f'policy_rate={self.policy_rate_hz:.1f}Hz, '
            f'executor_rate={self.executor_rate_hz:.1f}Hz, '
            f'substeps={self.substeps}, '
            f'interpolation={self.interpolation}, '
            f'gripper_mode={self.gripper_mode}'
        )

    def stop(self, drain=False, drain_timeout=None):
        if self.thread is None:
            return True

        drained = True
        if drain and self.thread.is_alive():
            drained = self.drain(timeout=drain_timeout)

        self.stop_event.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            rospy.logerr(
                f'{self.log_prefix} Stop requested but executor thread did not exit within 2.0s. '
                'It may still publish commands; skipping any direct homing publish is recommended.'
            )
            return False

        self.thread = None
        rospy.loginfo(f'{self.log_prefix} Stopped')
        return drained

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def reset(self, initial_action=None):
        with self.lock:
            self.waypoint_buffer.clear()
            self.anchor_waypoint = None if initial_action is None else self._validate_action(initial_action).copy()
            self.segment_start = None
            self.segment_target = None
            self.segment_step = 0
            self.loop_active = False

    def enqueue(self, action):
        """Compatibility alias used by older client code."""
        self.commit_action(action)

    def commit_action(self, action):
        action = self._validate_action(action)

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            with self.lock:
                if len(self.waypoint_buffer) < self.max_queue_size:
                    self.waypoint_buffer.append(action.copy())
                    return

            rospy.logwarn_throttle(
                1.0,
                f'{self.log_prefix} Waypoint buffer full ({self.max_queue_size}); '
                'waiting for trajectory player to catch up without dropping standard-mode waypoints.',
            )
            time.sleep(0.1)

    def commit_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            self.commit_action(actions)
            return

        for action in actions:
            self.commit_action(action)

    def merge_rtc_action(self, action, committed_prefix):
        self.merge_rtc_actions([action], committed_prefix=committed_prefix)

    def merge_rtc_actions(self, actions, committed_prefix):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        validated_actions = [self._validate_action(action).copy() for action in actions]
        committed_prefix = max(0, int(committed_prefix))

        with self.lock:
            preserved_count = min(committed_prefix, len(self.waypoint_buffer))
            preserved_waypoints = [self.waypoint_buffer[index] for index in range(preserved_count)]
            self.waypoint_buffer.clear()
            self.waypoint_buffer.extend(preserved_waypoints)

            available_slots = max(0, self.max_queue_size - len(self.waypoint_buffer))
            for action in validated_actions[:available_slots]:
                self.waypoint_buffer.append(action)

            if len(validated_actions) > available_slots:
                rospy.logwarn_throttle(
                    1.0,
                    f'{self.log_prefix} RTC tail truncated because waypoint buffer is full: '
                    f'accepted={available_slots}, requested={len(validated_actions)}',
                )

    def drain(self, timeout=None):
        if timeout is None:
            timeout = self._default_drain_timeout()

        deadline = time.time() + float(timeout)
        sleep_s = min(max(1.0 / self.executor_rate_hz, 0.001), 0.05)

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            with self.lock:
                segment_done = self.segment_target is None or self.segment_step >= self.substeps
                drained = not self.waypoint_buffer and segment_done and not self.loop_active

            if drained:
                return True

            if time.time() >= deadline:
                rospy.logwarn(
                    f'{self.log_prefix} Timed out waiting for trajectory drain after {timeout:.2f}s. '
                    'Stopping with pending waypoints may skip the last endpoint.'
                )
                return False

            time.sleep(sleep_s)

        return False

    def _default_drain_timeout(self):
        queue_s = float(self.max_queue_size) / max(self.policy_rate_hz, 1e-6)
        segment_s = float(self.substeps + 1) / max(self.executor_rate_hz, 1e-6)
        return max(2.0, queue_s + segment_s + 1.0)

    def _validate_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (14,):
            raise ValueError(f'Expected action shape (14,), got {action.shape}')
        if not np.all(np.isfinite(action)):
            raise ValueError(f'Action contains NaN/Inf: {action}')
        return action

    def _run_loop(self):
        rate = rospy.Rate(self.executor_rate_hz)
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            try:
                with self.lock:
                    self.loop_active = True
                cmd = self._next_command()
                if cmd is not None:
                    self._publish(cmd)
            except Exception as exc:
                rospy.logerr(f'{self.log_prefix} Error in trajectory loop: {exc}')
            finally:
                with self.lock:
                    self.loop_active = False
            rate.sleep()

    def _next_command(self):
        with self.lock:
            if self.anchor_waypoint is None:
                waypoint = self._pop_next_waypoint_locked()
                if waypoint is None:
                    return None
                self.anchor_waypoint = waypoint.copy()
                return self.anchor_waypoint.copy()

            if self.segment_target is None or self.segment_step >= self.substeps:
                waypoint = self._pop_next_waypoint_locked()
                if waypoint is None:
                    return self.anchor_waypoint.copy()

                self.segment_start = self.anchor_waypoint.copy()
                self.segment_target = self._unwrap_target(self.segment_start, waypoint)
                self.segment_step = 0

            self.segment_step += 1
            u = float(self.segment_step) / float(self.substeps)
            s = self._interp_scalar(u)
            cmd = self.segment_start + s * (self.segment_target - self.segment_start)

            if self.gripper_mode == 'passthrough':
                for index in self.GRIPPER_INDICES:
                    cmd[index] = self.segment_target[index]

            if self.segment_step >= self.substeps:
                cmd = self.segment_target.copy()
                self.anchor_waypoint = self.segment_target.copy()

            return cmd.copy()

    def _pop_next_waypoint_locked(self):
        qsize = len(self.waypoint_buffer)
        if qsize > self.max_queue_size * 0.5:
            rospy.logwarn_throttle(
                1.0,
                f'{self.log_prefix} Waypoint backlog: {qsize}/{self.max_queue_size}',
            )

        if not self.waypoint_buffer:
            return None
        return self.waypoint_buffer.popleft()

    def _interp_scalar(self, u):
        u = min(max(float(u), 0.0), 1.0)
        if self.interpolation == 'linear':
            return u
        if self.interpolation == 'minimum_jerk':
            return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        raise ValueError(f'Unknown interpolation: {self.interpolation}')

    def _unwrap_target(self, start, target):
        target = target.copy()
        for index in self.RPY_INDICES:
            target[index] = self._shortest_angle_target(start[index], target[index])
        return target

    @staticmethod
    def _shortest_angle_target(start_angle, target_angle):
        delta = (target_angle - start_angle + np.pi) % (2.0 * np.pi) - np.pi
        return start_angle + delta

    def _publish(self, cmd):
        self.ros_operator.eef_arm_publish(cmd[:7], cmd[7:14])
