#!/usr/bin/env python3
"""msgpack numpy roundtrip tests for the robot-side client packer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CLIENT_ROOT = Path(__file__).resolve().parents[1] / 'client'
sys.path.insert(0, str(CLIENT_ROOT))

from dwam_client.msgpack_numpy import packb, unpackb  # noqa: E402


class TestMsgpackNumpy(unittest.TestCase):
    def test_ndarray_roundtrip(self):
        original = {
            'action': np.arange(16 * 2 * 16, dtype=np.float32).reshape(16, 2, 16),
            'reset': False,
            'prompt': 'open the toolbox',
        }
        restored = unpackb(packb(original))
        self.assertEqual(restored['prompt'], original['prompt'])
        self.assertFalse(restored['reset'])
        np.testing.assert_array_equal(restored['action'], original['action'])

    def test_image_batch(self):
        frame = {
            'observation.images.cam_high': np.zeros((256, 320, 3), dtype=np.uint8),
            'observation.images.cam_left_wrist': np.ones((128, 160, 3), dtype=np.uint8),
        }
        restored = unpackb(packb({'obs': [frame]}))
        self.assertEqual(restored['obs'][0]['observation.images.cam_high'].shape, (256, 320, 3))
        self.assertEqual(restored['obs'][0]['observation.images.cam_left_wrist'].dtype, np.uint8)


if __name__ == '__main__':
    unittest.main()
