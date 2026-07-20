#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS Operator for OpenPI LIFT2 Client (EEF Control)
Handles sensor data collection and end-effector pose control for ARX R5 dual-arm robot
"""

import os
import queue
import threading
import time

import cv2
import rospy
from sensor_msgs.msg import Image
from arm_control.msg import PosCmd
from cv_bridge import CvBridge
from collections import deque
import numpy as np


def save_recorded_videos_from_frames(pic_output_dir, video_output_dir, camera_file_names, camera_pic_dir_names, fps=60.0, log_fn=None):
    logger = log_fn or (lambda message: None)

    for camera_name, file_name in camera_file_names.items():
        pic_dir = os.path.join(pic_output_dir, camera_pic_dir_names[camera_name])
        if not os.path.isdir(pic_dir):
            logger(f"[Video] Missing frame directory for {camera_name}, skipping {file_name}")
            continue

        frame_names = sorted(
            name for name in os.listdir(pic_dir)
            if name.lower().endswith('.jpg')
        )
        frame_count = len(frame_names)
        if frame_count == 0:
            logger(f"[Video] No frames recorded for {camera_name}, skipping {file_name}")
            continue

        first_frame_path = os.path.join(pic_dir, frame_names[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            raise RuntimeError(f"Failed to read first frame from {first_frame_path}")

        height, width = first_frame.shape[:2]
        file_path = os.path.join(video_output_dir, file_name)
        writer = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {file_path}")

        try:
            for frame_name in frame_names:
                frame_path = os.path.join(pic_dir, frame_name)
                frame = cv2.imread(frame_path)
                if frame is None:
                    raise RuntimeError(f"Failed to read frame from {frame_path}")
                writer.write(frame)
        finally:
            writer.release()

        logger(f"[Video] Saved {camera_name}: {frame_count} frames -> {file_path} @ {fps:.1f} FPS")


class RosOperator:
    """ROS Operator: Manages all ROS topic subscriptions and publications"""

    CAMERA_FILE_NAMES = {
        'head': 'camera_h.mp4',
        'left_wrist': 'camera_l.mp4',
        'right_wrist': 'camera_r.mp4',
    }

    CAMERA_PIC_DIR_NAMES = {
        'head': 'camera_h',
        'left_wrist': 'camera_l',
        'right_wrist': 'camera_r',
    }

    def __init__(self, args):
        """
        Initialize ROS operator

        Args:
            args: Command line arguments containing all topic names
        """
        self.args = args
        self.bridge = CvBridge()

        # Initialize data queues
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()

        # End-effector pose queues
        self.arm_left_pose_deque = deque()
        self.arm_right_pose_deque = deque()

        # Publishers
        self.arm_left_cmd_publisher = None
        self.arm_right_cmd_publisher = None

        self.video_enabled = bool(getattr(self.args, 'video_output_dir', None))
        self.camera_file_names = getattr(self.args, 'camera_file_names', None) or self.CAMERA_FILE_NAMES
        self.camera_pic_dir_names = getattr(self.args, 'camera_pic_dir_names', None) or self.CAMERA_PIC_DIR_NAMES
        self.recorded_frame_count = 0
        self.recorded_frame_counts = {
            'head': 0,
            'left_wrist': 0,
            'right_wrist': 0,
        }

        # Recording thread
        self.recording_thread = None
        self.recording_active = False
        self.recording_stop_event = threading.Event()
        self.recording_lock = threading.Lock()
        self.frame_save_queue = queue.Queue(maxsize=240)
        self.frame_save_thread = None
        self.frame_save_active = False
        self.frame_save_stop_event = threading.Event()
        self.recording_loop_hz = 60.0
        self.recording_started_wall_s = None
        self.recording_last_status_log_s = 0.0

        # Initialize ROS topics
        self.init_ros()

    def init_ros(self):
        """Initialize ROS subscribers and publishers"""
        # Note: rospy.init_node is called in main(), not here

        if self.video_enabled:
            self.init_video_recording()

        # ========== Subscribe to camera topics ==========
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback,
                        queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback,
                        queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback,
                        queue_size=1000, tcp_nodelay=True)

        if hasattr(self.args, 'use_depth_image') and self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback,
                            queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback,
                            queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback,
                            queue_size=1000, tcp_nodelay=True)

        # ========== Subscribe to arm end-effector pose topics ==========
        rospy.Subscriber(self.args.arm_left_pose_topic, PosCmd,
                        self.arm_left_pose_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.arm_right_pose_topic, PosCmd,
                        self.arm_right_pose_callback, queue_size=1000, tcp_nodelay=True)

        # ========== Create publishers (control commands) ==========
        self.arm_left_cmd_publisher = rospy.Publisher(self.args.arm_left_cmd_topic,
                                                      PosCmd, queue_size=10)
        self.arm_right_cmd_publisher = rospy.Publisher(self.args.arm_right_cmd_topic,
                                                       PosCmd, queue_size=10)

        rospy.loginfo("ROS Operator initialized (EEF Control)")

    def init_video_recording(self):
        os.makedirs(self.args.pic_output_dir, exist_ok=True)
        os.makedirs(self.args.video_output_dir, exist_ok=True)

        self.recorded_frame_count = 0
        for camera_name in self.camera_file_names:
            pic_dir = self._get_camera_pic_dir(camera_name)
            os.makedirs(pic_dir, exist_ok=True)
            self.recorded_frame_counts[camera_name] = 0
            rospy.loginfo(f"[Video] Ready to collect frames for {camera_name} -> {pic_dir}")

    def _get_camera_pic_dir(self, camera_name):
        return os.path.join(self.args.pic_output_dir, self.camera_pic_dir_names[camera_name])

    def start_recording(self):
        """Start the recording thread"""
        if not self.video_enabled or self.recording_active:
            return

        self._start_frame_save_worker()
        self.recording_active = True
        self.recording_stop_event.clear()
        self.recording_started_wall_s = time.time()
        self.recording_last_status_log_s = self.recording_started_wall_s
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()
        rospy.loginfo("[Recording] Started recording thread at 60Hz")

    def stop_recording(self):
        """Stop the recording thread"""
        if not self.recording_active:
            return

        self.recording_stop_event.set()
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)
        self.recording_active = False
        self._stop_frame_save_worker()
        elapsed_s = max(time.time() - (self.recording_started_wall_s or time.time()), 1e-6)
        effective_hz = self.recorded_frame_count / elapsed_s
        rospy.loginfo(
            f"[Recording] Stopped. Recorded frames: head={self.recorded_frame_counts['head']}, "
            f"left={self.recorded_frame_counts['left_wrist']}, right={self.recorded_frame_counts['right_wrist']}, "
            f"elapsed={elapsed_s:.2f}s, effective_save_hz={effective_hz:.2f}"
        )

    def _start_frame_save_worker(self):
        if self.frame_save_active:
            return
        self.frame_save_active = True
        self.frame_save_stop_event.clear()
        self.frame_save_thread = threading.Thread(target=self._frame_save_worker_loop, daemon=True)
        self.frame_save_thread.start()

    def _stop_frame_save_worker(self):
        if not self.frame_save_active:
            return
        self.frame_save_stop_event.set()
        try:
            self.frame_save_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.frame_save_thread:
            self.frame_save_thread.join(timeout=10.0)
        self.frame_save_active = False
        while not self.frame_save_queue.empty():
            try:
                self.frame_save_queue.get_nowait()
            except queue.Empty:
                break

    def _frame_save_worker_loop(self):
        while not self.frame_save_stop_event.is_set() and not rospy.is_shutdown():
            try:
                item = self.frame_save_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            frame_index, frames = item
            try:
                self._write_frame_set_to_disk(frame_index, frames)
            except Exception as exc:
                rospy.logwarn(f"[Recording] Async frame save error at index {frame_index}: {exc}")

    def _write_frame_set_to_disk(self, frame_index, frames):
        for camera_name, frame_bgr in frames.items():
            frame_path = os.path.join(self._get_camera_pic_dir(camera_name), f"{frame_index:06d}.jpg")
            if not cv2.imwrite(frame_path, frame_bgr):
                raise RuntimeError(f"Failed to save frame to {frame_path}")

        with self.recording_lock:
            self.recorded_frame_count = frame_index + 1
            for camera_name in self.recorded_frame_counts:
                self.recorded_frame_counts[camera_name] = self.recorded_frame_count

    def _maybe_log_recording_status(self):
        now_s = time.time()
        if now_s - self.recording_last_status_log_s < 5.0:
            return
        self.recording_last_status_log_s = now_s
        elapsed_s = max(now_s - (self.recording_started_wall_s or now_s), 1e-6)
        effective_hz = self.recorded_frame_count / elapsed_s
        queue_depth = self.frame_save_queue.qsize()
        rospy.loginfo(
            f"[Recording] status: saved_frames={self.recorded_frame_count}, "
            f"elapsed={elapsed_s:.1f}s, effective_save_hz={effective_hz:.2f}, "
            f"save_queue_depth={queue_depth}"
        )

    def _recording_loop(self):
        """Recording thread main loop - runs at 60Hz independently"""
        rate = rospy.Rate(self.recording_loop_hz)

        while not self.recording_stop_event.is_set() and not rospy.is_shutdown():
            if (len(self.img_front_deque) > 0 and
                len(self.img_left_deque) > 0 and
                len(self.img_right_deque) > 0):

                try:
                    img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque[-1], 'passthrough')
                    img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque[-1], 'passthrough')
                    img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque[-1], 'passthrough')
                    self._enqueue_current_frames(img_front, img_left, img_right)
                except Exception as e:
                    rospy.logwarn(f"[Recording] Frame capture error: {e}")

            self._maybe_log_recording_status()
            rate.sleep()

    def _enqueue_current_frames(self, img_front, img_left, img_right):
        if not self.video_enabled:
            return
        if img_front is None or img_left is None or img_right is None:
            return

        with self.recording_lock:
            frame_index = self.recorded_frame_count + self.frame_save_queue.qsize()

        frames = {
            'head': cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR),
            'left_wrist': cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR),
            'right_wrist': cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR),
        }

        try:
            self.frame_save_queue.put_nowait((frame_index, frames))
        except queue.Full:
            rospy.logwarn_throttle(5.0, "[Recording] Frame save queue full; dropping frame to preserve 60Hz capture")

    def save_current_frames_to_disk(self, img_front, img_left, img_right):
        """Save one synchronized frame set to disk."""
        if not self.video_enabled:
            return
        if img_front is None or img_left is None or img_right is None:
            return

        frames = {
            'head': cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR),
            'left_wrist': cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR),
            'right_wrist': cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR),
        }

        with self.recording_lock:
            frame_index = self.recorded_frame_count
        self._write_frame_set_to_disk(frame_index, frames)

    def save_recorded_videos(self):
        if not self.video_enabled:
            return

        save_recorded_videos_from_frames(
            self.args.pic_output_dir,
            self.args.video_output_dir,
            self.camera_file_names,
            self.camera_pic_dir_names,
            fps=60.0,
            log_fn=rospy.loginfo,
        )

    def close_video_writers(self):
        if self.recording_active:
            self.stop_recording()

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    # ==================== Arm end-effector pose callbacks ====================
    def arm_left_pose_callback(self, msg):
        """
        Callback for left arm end-effector pose

        Args:
            msg: PosCmd message with x, y, z, roll, pitch, yaw, gripper
        """
        if len(self.arm_left_pose_deque) >= 2000:
            self.arm_left_pose_deque.popleft()
        self.arm_left_pose_deque.append(msg)

    def arm_right_pose_callback(self, msg):
        """
        Callback for right arm end-effector pose

        Args:
            msg: PosCmd message with x, y, z, roll, pitch, yaw, gripper
        """
        if len(self.arm_right_pose_deque) >= 2000:
            self.arm_right_pose_deque.popleft()
        self.arm_right_pose_deque.append(msg)

    # ==================== Synchronized data acquisition ====================
    def get_frame(self):
        """
        Get all sensor data with timestamp synchronization

        Returns:
            tuple: (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                    arm_left_pose, arm_right_pose)
            False: Data not ready or timestamps not aligned
        """
        # Check if basic data is ready
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0:
            return False

        if hasattr(self.args, 'use_depth_image') and self.args.use_depth_image:
            if len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0:
                return False

        if len(self.arm_left_pose_deque) == 0 or len(self.arm_right_pose_deque) == 0:
            return False

        # Get minimum timestamp
        timestamps = [
            self.img_left_deque[-1].header.stamp.to_sec(),
            self.img_right_deque[-1].header.stamp.to_sec(),
            self.img_front_deque[-1].header.stamp.to_sec(),
        ]

        if hasattr(self.args, 'use_depth_image') and self.args.use_depth_image:
            timestamps.extend([
                self.img_left_depth_deque[-1].header.stamp.to_sec(),
                self.img_right_depth_deque[-1].header.stamp.to_sec(),
                self.img_front_depth_deque[-1].header.stamp.to_sec(),
            ])

        frame_time = min(timestamps)

        # Check if all data has reached this timestamp
        if self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False

        # Pop old data and get synchronized data
        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        # Depth images (optional)
        img_left_depth = None
        img_right_depth = None
        img_front_depth = None
        if hasattr(self.args, 'use_depth_image') and self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        # End-effector poses (take latest)
        arm_left_pose = self.arm_left_pose_deque[-1]
        arm_right_pose = self.arm_right_pose_deque[-1]

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                arm_left_pose, arm_right_pose)

    def get_latest_images(self):
        """Peek the latest camera frames without popping queues.

        Used for VA keyframe sampling during action execution so the control
        loop does not drain the image deques via ``get_frame()``.

        Returns:
            tuple: (img_front, img_left, img_right) as RGB uint8 (passthrough),
            or False if any camera queue is empty.
        """
        if (
            len(self.img_left_deque) == 0
            or len(self.img_right_deque) == 0
            or len(self.img_front_deque) == 0
        ):
            return False

        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque[-1], 'passthrough')
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque[-1], 'passthrough')
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque[-1], 'passthrough')
        return (img_front, img_left, img_right)

    def get_latest_arm_poses(self):
        """Peek latest dual-arm EEF poses without popping."""
        if len(self.arm_left_pose_deque) == 0 or len(self.arm_right_pose_deque) == 0:
            return False
        return (self.arm_left_pose_deque[-1], self.arm_right_pose_deque[-1])

    # ==================== Action publishing ====================
    def eef_arm_publish(self, left, right):
        """
        Publish dual-arm end-effector pose control commands

        Args:
            left: (7,) or list - Left arm pose [x, y, z, roll, pitch, yaw, gripper]
            right: (7,) or list - Right arm pose [x, y, z, roll, pitch, yaw, gripper]
        """
        # Left arm
        pos_cmd_left = PosCmd()
        pos_cmd_left.x = left[0]
        pos_cmd_left.y = left[1]
        pos_cmd_left.z = left[2]
        pos_cmd_left.roll = left[3]
        pos_cmd_left.pitch = left[4]
        pos_cmd_left.yaw = left[5]
        pos_cmd_left.gripper = left[6]

        # Right arm
        pos_cmd_right = PosCmd()
        pos_cmd_right.x = right[0]
        pos_cmd_right.y = right[1]
        pos_cmd_right.z = right[2]
        pos_cmd_right.roll = right[3]
        pos_cmd_right.pitch = right[4]
        pos_cmd_right.yaw = right[5]
        pos_cmd_right.gripper = right[6]

        # Publish
        self.arm_left_cmd_publisher.publish(pos_cmd_left)
        self.arm_right_cmd_publisher.publish(pos_cmd_right)
