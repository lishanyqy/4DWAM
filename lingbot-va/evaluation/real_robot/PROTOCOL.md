# real_robot VA Protocol

Authoritative I/O contract for [`evaluation/real_robot`](./).

## Roles

| Side | Code | Aligns with |
|------|------|-------------|
| GPU server | `wan_va.wan_va_server` + thin infer cfg | robotwin / LingBot-VA |
| Robot client | `client/deploy/client_lift2_va.py` | openpi-on-LIFT2 ROS + this protocol |

Default WebSocket port: **7777**.

Default robot publish rate: **60 Hz**.

## Transport

- WebSocket + msgpack with numpy extension (`msgpack_numpy`)
- `compression=None`
- Client `ping_interval=None` (long diffusion steps)
- First server message after connect: metadata dict (may be empty `{}`)

## Server requests

### 1. reset

```python
{"reset": True, "prompt": "<language instruction>"}
# response: {}
```

### 2. infer (action chunk)

```python
{"obs": <frame_dict | list[frame_dict]>}
# response: {"action": ndarray, "server_timing": optional}
```

`VA_Server._encode_obs` reads **`payload["obs"]`**.

Frame keys (robotwin_tshape):

```text
observation.images.cam_high          # uint8 RGB, tshape: 256x320
observation.images.cam_left_wrist    # uint8 RGB, tshape: 128x160
observation.images.cam_right_wrist   # uint8 RGB, tshape: 128x160
```

### 3. compute_kv_cache

```python
{
  "compute_kv_cache": True,
  "obs": [frame_dict, ...],   # keyframes collected during execution
  "state": action_chunk,      # previous infer `action` array (C,F,H)
}
# response: {}
```

## Action tensor

| Item | Value |
|------|-------|
| Field name | `action` (not openpi `actions`) |
| Typical shape | `(C, F, H) = (16, 2, 16)` |
| `C` | used action channels (xyz+quat+grip × 2) |
| `F` | `frame_chunk_size` |
| `H` | `action_per_frame` |
| Pose semantics | **segment-relative** until client re-anchors |
| Grip | ~`[0, 1]` on policy side |

## Client bridge (segment_relative_va)

**Not** openpi delta. Current conversion:

```text
server action[:,i,j]  (16D relative xyz+quat_xyzw+grip)
  → add_init_pose(rel, episode_init_16d)   # abs = init ⊕ relative
  → eef16_to_eef14                         # quat_xyzw → rpy for PosCmd
  → denormalize_gripper [0,1] → [0,5]
  → RosOperator.eef_arm_publish
```

1. Cache episode `init_eef` as 16D (`pose_to_eef` 14D → quat **xyzw**, SciPy / training).
2. For each micro-step `action[:, i, j]`:
   - `C==16`: `add_init_pose` → absolute 16D → 14D euler
   - grip: `[0,1] → [0, gripper_raw_max]` (default 5)
3. First chunk: execute `i` from `first_chunk_start_idx` (default **1**).
4. Keyframe when `(j+1) % (H // keyframe_divisor) == 0` (default divisor **4**).
5. Keyframes: **peek** latest cameras (`get_latest_images`), do **not** pop queues.
6. Empty `key_frames` → **raise** before `compute_kv_cache` (fail loud).
7. After chunk: `compute_kv_cache(keyframes, state=full action chunk)` (same as sim).

Do **not** re-anchor every step with the latest robot state.

## Images

RosOperator / openpi real robot: in-memory frames are **RGB** (`passthrough`; recording uses `RGB2BGR` only for disk). Client must **not** BGR-swap before VA.

## Publish rate sleep

Between micro-steps the client sleeps so wall-clock spacing ≈ `1/publish_rate` (default 60 Hz). Without sleep, PosCmd would be spammed as fast as CPU allows.

## ROS I/O (openpi-on-LIFT2 aligned)

**Subscribe**

- `/camera_h|l|r/color/image_raw`
- `/arm_left|right/arm_status_ee` (`arm_control/PosCmd`)

**Publish**

- `/arm_left_cmd`, `/arm_right_cmd` absolute `PosCmd`
- 14D: `[L_xyz, L_rpy, L_grip_raw, R_xyz, R_rpy, R_grip_raw]`

## Episode loop

```text
reset(prompt)
cache init_eef
loop:
  infer(obs=first_obs) -> action (C,F,H)
  for micro-step in expand(action):
    publish absolute 14D @ publish_rate (default 60 Hz)
    maybe sample keyframe
  compute_kv_cache(key_frames, state=action)
```

## Infer configs

| Name | Purpose |
|------|---------|
| `lift2_merged_infer` | LingBot-VA LIFT2 checkpoint |
| `4dwam_lift2_infer` | 4DWAM LIFT2 checkpoint (same protocol) |

Both set `infer_mode=server`, `port=7777`, `env_type=robotwin_tshape`, and load `norm_stat` from the LIFT2 dataset stats file.
