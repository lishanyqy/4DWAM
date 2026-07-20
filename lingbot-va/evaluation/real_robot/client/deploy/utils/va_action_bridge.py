#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VA action chunk bridge: (C, F, H) segment-relative -> absolute 14D EEF queue.

Matches robotwin ``eval_polict_client_openpi.py`` execution:
  - first chunk skips frame index 0 (start_idx=1)
  - C==16: add_init_pose re-anchor, then convert to 14D euler for ROS
  - keyframes every (H // keyframe_divisor) micro-steps on each executed F index
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .rotation import (
    add_init_pose,
    denormalize_gripper,
    eef14_to_eef16,
    eef16_to_eef14,
)


@dataclass(frozen=True)
class MicroStep:
    """One executable micro action with bookkeeping for keyframe sampling."""

    absolute_eef14_policy: np.ndarray  # grip still [0, 1]
    absolute_eef14_raw: np.ndarray  # grip mapped to robot raw range
    frame_index: int
    micro_index: int
    is_keyframe: bool


def count_keyframes(
    frame_count: int,
    action_per_frame: int,
    start_idx: int,
    keyframe_divisor: int = 4,
) -> int:
    if action_per_frame % keyframe_divisor != 0:
        raise ValueError(
            f'action_per_frame={action_per_frame} must be divisible by '
            f'keyframe_divisor={keyframe_divisor}'
        )
    executed_frames = frame_count - start_idx
    if executed_frames < 0:
        raise ValueError(f'Invalid start_idx={start_idx} for F={frame_count}')
    return executed_frames * (action_per_frame // keyframe_divisor)


def expand_action_chunk(
    action: np.ndarray,
    init_eef16: np.ndarray,
    *,
    first_chunk: bool = False,
    first_chunk_start_idx: int = 1,
    keyframe_divisor: int = 4,
    gripper_raw_max: float = 5.0,
    bridge_mode: str = 'segment_relative_va',
) -> List[MicroStep]:
    """Expand server ``action`` into ordered absolute 14D micro-steps."""
    action = np.asarray(action)
    if action.ndim != 3:
        raise ValueError(
            f'Expected action shape (C, F, H), got {action.shape}'
        )
    channel_count, frame_count, action_per_frame = action.shape
    if action_per_frame % keyframe_divisor != 0:
        raise ValueError(
            f'H={action_per_frame} must be divisible by keyframe_divisor='
            f'{keyframe_divisor}'
        )

    start_idx = int(first_chunk_start_idx) if first_chunk else 0
    if start_idx < 0 or start_idx > frame_count:
        raise ValueError(f'Invalid start_idx={start_idx} for F={frame_count}')

    mode = (bridge_mode or 'segment_relative_va').lower()
    if mode not in ('segment_relative_va', 'absolute_stored'):
        raise ValueError(
            f'Unsupported bridge_mode={bridge_mode!r}. '
            'Use segment_relative_va (default) or absolute_stored.'
        )

    init_eef16 = np.asarray(init_eef16, dtype=np.float64).reshape(16)
    stride = action_per_frame // keyframe_divisor
    steps: List[MicroStep] = []

    for frame_index in range(start_idx, frame_count):
        for micro_index in range(action_per_frame):
            ee = action[:, frame_index, micro_index].astype(np.float64)

            if channel_count == 16:
                if mode == 'segment_relative_va':
                    absolute_16 = add_init_pose(ee, init_eef16)
                else:
                    absolute_16 = ee.copy()
                    left_norm = np.linalg.norm(absolute_16[3:7])
                    right_norm = np.linalg.norm(absolute_16[11:15])
                    absolute_16[3:7] = absolute_16[3:7] / max(left_norm, 1e-8)
                    absolute_16[11:15] = absolute_16[11:15] / max(right_norm, 1e-8)
                absolute_14_policy = eef16_to_eef14(absolute_16)
            elif channel_count == 14:
                # Compatibility branch: treat as absolute euler EEF.
                absolute_14_policy = ee.astype(np.float64).reshape(14)
            else:
                raise NotImplementedError(
                    f'Unsupported action channel count C={channel_count}; '
                    'expected 16 (preferred) or 14'
                )

            absolute_14_raw = absolute_14_policy.copy()
            absolute_14_raw[6] = denormalize_gripper(
                absolute_14_policy[6],
                gripper_max=gripper_raw_max,
            )
            absolute_14_raw[13] = denormalize_gripper(
                absolute_14_policy[13],
                gripper_max=gripper_raw_max,
            )

            is_keyframe = ((micro_index + 1) % stride) == 0
            steps.append(
                MicroStep(
                    absolute_eef14_policy=absolute_14_policy.astype(np.float32),
                    absolute_eef14_raw=absolute_14_raw.astype(np.float32),
                    frame_index=frame_index,
                    micro_index=micro_index,
                    is_keyframe=is_keyframe,
                )
            )

    return steps


class VAActionBridge:
    """Stateful helper: first-chunk start_idx + expand + init pose cache."""

    def __init__(
        self,
        *,
        first_chunk_start_idx: int = 1,
        keyframe_divisor: int = 4,
        gripper_raw_max: float = 5.0,
        bridge_mode: str = 'segment_relative_va',
    ) -> None:
        self.first_chunk_start_idx = int(first_chunk_start_idx)
        self.keyframe_divisor = int(keyframe_divisor)
        self.gripper_raw_max = float(gripper_raw_max)
        self.bridge_mode = bridge_mode
        self._init_eef16: Optional[np.ndarray] = None
        self._chunk_index = 0

    def reset(self, init_eef14_or_16: np.ndarray) -> None:
        init = np.asarray(init_eef14_or_16, dtype=np.float64).reshape(-1)
        if init.shape[0] == 14:
            self._init_eef16 = eef14_to_eef16(init)
        elif init.shape[0] == 16:
            self._init_eef16 = init.astype(np.float64)
        else:
            raise ValueError(
                f'init EEF must be 14D or 16D, got shape {init.shape}'
            )
        self._chunk_index = 0

    @property
    def init_eef16(self) -> np.ndarray:
        if self._init_eef16 is None:
            raise RuntimeError('VAActionBridge.reset(init_eef) must be called first')
        return self._init_eef16

    def expand(self, action: np.ndarray) -> List[MicroStep]:
        first_chunk = self._chunk_index == 0
        steps = expand_action_chunk(
            action,
            self.init_eef16,
            first_chunk=first_chunk,
            first_chunk_start_idx=self.first_chunk_start_idx,
            keyframe_divisor=self.keyframe_divisor,
            gripper_raw_max=self.gripper_raw_max,
            bridge_mode=self.bridge_mode,
        )
        self._chunk_index += 1
        return steps

    def iter_steps(self, action: np.ndarray) -> Iterator[MicroStep]:
        yield from self.expand(action)
