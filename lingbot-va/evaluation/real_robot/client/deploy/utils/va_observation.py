#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS / raw camera frames -> VA observation dict (robotwin_tshape keys).

Color convention (aligned with openpi-on-LIFT2 real robot):
  RosOperator uses ``imgmsg_to_cv2(..., 'passthrough')`` and openpi feeds that
  array straight into policy without BGR/RGB conversion. Recording code in
  openpi explicitly does ``cv2.cvtColor(..., COLOR_RGB2BGR)`` before imwrite,
  which proves the in-memory pipeline is **RGB**.

  This client therefore treats RosOperator images as **RGB uint8** by default.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

VA_CAM_HIGH = 'observation.images.cam_high'
VA_CAM_LEFT = 'observation.images.cam_left_wrist'
VA_CAM_RIGHT = 'observation.images.cam_right_wrist'

# robotwin_tshape defaults used by VA_Server._encode_obs
TSHAPE_HIGH_HW = (256, 320)  # H, W
TSHAPE_WRIST_HW = (128, 160)


def ensure_uint8_rgb(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f'Expected HxWx3/4 image, got shape {image.shape}')
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        if image.max() <= 1.0 + 1e-6:
            image = image * 255.0
        image = image.astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return np.ascontiguousarray(image)


def resize_rgb(image_rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    image = ensure_uint8_rgb(image_rgb)
    if image.shape[0] == height and image.shape[1] == width:
        return image
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized)


def build_va_frame(
    head_rgb: np.ndarray,
    left_wrist_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
    camera_mode: str = 'tshape',
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one VA observation frame from **RGB** images (openpi convention).

    Args:
        head_rgb / left_wrist_rgb / right_wrist_rgb: HxWx3 RGB from RosOperator.
        camera_mode: ``tshape`` (default LIFT2/robotwin) or ``native`` (no resize).
        task: optional language string stored on the frame (server uses reset prompt).
    """
    head = ensure_uint8_rgb(head_rgb)
    left = ensure_uint8_rgb(left_wrist_rgb)
    right = ensure_uint8_rgb(right_wrist_rgb)

    mode = (camera_mode or 'tshape').lower()
    if mode == 'tshape':
        high_h, high_w = TSHAPE_HIGH_HW
        wrist_h, wrist_w = TSHAPE_WRIST_HW
        head = resize_rgb(head, high_h, high_w)
        left = resize_rgb(left, wrist_h, wrist_w)
        right = resize_rgb(right, wrist_h, wrist_w)
    elif mode == 'native':
        pass
    else:
        raise ValueError(f'Unknown camera_mode={camera_mode!r}')

    frame: Dict[str, Any] = {
        VA_CAM_HIGH: head,
        VA_CAM_LEFT: left,
        VA_CAM_RIGHT: right,
    }
    if task is not None:
        frame['task'] = task
    return frame


def wrap_obs_for_server(frame_or_list) -> Dict[str, Any]:
    """Wrap frames so ``VA_Server._encode_obs`` sees ``payload['obs']``."""
    return {'obs': frame_or_list}


def tshape_sizes() -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return TSHAPE_HIGH_HW, TSHAPE_WRIST_HW
