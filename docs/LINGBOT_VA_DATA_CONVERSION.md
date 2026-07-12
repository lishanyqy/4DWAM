# LIFT2 HDF5 to LingBot-VA / 4DWAM data preparation

This document defines the current LIFT2 conversion boundary for the local
Robbyant LingBot-VA + 4DWAM codebase. It intentionally distinguishes two media
layouts, because the restored upstream Step 3 script and standard LeRobot video
storage do not use the same raw-camera representation.

The converter is:

```text
lingbot-va/preprocess/convert_hdf5_to_lingbot_lerobot.py
```

The output is only the raw LeRobot stage. LingBot-VA / 4DWAM training also
requires latent `.pth` artifacts, trace `.pth` artifacts when trace alignment is
action representation.

## 1. Decisions and non-negotiable contracts

### 1.1 Keep the original 60 Hz LIFT2 timeline

The raw LIFT2 recordings are treated as 60 Hz. The converter requires:

```text
source_fps == fps == 60
```

It deliberately refuses to:

- downsample from 60 Hz to 30 Hz;
- upsample with blended RGB frames or linearly interpolated Euler angles;
- remove a static prefix;
- modify episode boundaries.

These transformations change the frame index space used by `action_config`,
latent `frame_ids`, TraceAnything extraction, and action supervision. If a
future data-cleaning stage removes frames, it must emit an explicit raw-frame to
converted-frame mapping and regenerate all downstream artifacts.

### 1.2 Preserve the raw action contract

The converter writes the following 14D absolute action from top-level HDF5
`action_eef`:

```text
[
  left_x, left_y, left_z,
  left_roll, left_pitch, left_yaw,
  left_gripper,
  right_x, right_y, right_z,
  right_roll, right_pitch, right_yaw,
  right_gripper,
]
```

It writes `observations/qpos` unchanged as the 14D
`observation.state` feature.

Do **not** precompute delta actions, relative poses, quaternions, or padded 30D
actions in the converter. `LatentLeRobotDataset` currently performs:

```text
14D Euler EEF action
  -> 16D quaternion EEF action
  -> relative pose inside each action_config segment
  -> 30D multi-embodiment channel alignment
  -> normalization
```

LIFT2 raw gripper commands normally live in `[0, 5]`. The default converter
mode normalizes only action channels 6 and 13 into `[0, 1]`. It does not modify
the qpos state. This is only a raw-data convention; LIFT2 normalization
statistics must still be recomputed after the loader's full action transform.

### 1.3 Camera order is semantic

The three required camera feature keys are fixed by
`wan_va/configs/va_robotwin_cfg.py`:

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
```

The loader concatenates the wrist views horizontally and the high view below
them. Do not rename, reorder, omit, or substitute these keys without changing
the loader and `obs_cam_keys` together.

### 1.4 action_config is required metadata

Every converted episode has one explicit full-episode segment:

```json
{
  "episode_index": 0,
  "tasks": ["Open the toolbox and find the wrench."],
  "length": 1582,
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": 1582,
      "action_text": "Open the toolbox and find the wrench.",
      "skill": ""
    }
  ]
}
```

The `start_frame` and `end_frame` values are in the unmodified 60 Hz converted
episode index space. They are not raw HDF5 file names, wall-clock timestamps, or
downsampled indices.

## 2. Required HDF5 input

Each source directory contains files named:

```text
episode_0.hdf5
episode_1.hdf5
...
```

Each episode must provide synchronized datasets:

```text
action_eef                               [T, 14] float32
observations/qpos                        [T, 14] float32
observations/images/head                 [T, N]  encoded JPEG or PNG bytes
observations/images/left_wrist           [T, N]  encoded JPEG or PNG bytes
observations/images/right_wrist          [T, N]  encoded JPEG or PNG bytes
```

The converter checks every sequence length, action/state dimension, finite
action/state value, decodability of each camera frame, and camera resolution
consistency. A conversion fails rather than silently dropping a bad frame.

## 3. Two intentional media layouts

Both layouts provide MP4 files that can be consumed by
`preprocess/extract_latents_from_pixels_adaptv1.py`. Choose the layout according
to whether parquet-embedded image bytes are also required.

### 3.1 step3-compatible layout (default)

Use this layout when parquet-embedded image bytes are required in addition to
the MP4 files used by `preprocess/extract_latents_from_pixels_adaptv1.py`.

```text
dataset_root/
├── meta/
│   ├── info.json                    # cameras have dtype=image
│   ├── episodes.jsonl               # contains action_config
│   └── lingbot_conversion.json
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet   # action, state, and image bytes
├── images/
│   └── observation.images.cam_high/
│       └── episode_000000/
│           └── frame_*.png
└── videos/
    └── chunk-000/
        └── observation.images.cam_high/
            └── episode_000000.mp4
```

This is an explicit compatibility bridge:

- Step 3 reads the delayed MP4 sidecars from `videos/`;
- Step 4 reads the same MP4 sidecars from `videos/`;
- `meta/info.json` truthfully remains `dtype=image` and `total_videos` is not
  patched to pretend that the camera feature is native video.

The converter encodes MP4 sidecars only after every image-backed episode has
been saved. This avoids violating LeRobot's writer assertions while recording.

Use this mode when downstream consumers still need image bytes in parquet. It
is not the desired long-term LeRobot storage representation.

### 3.2 standard-video layout

Use this layout for a canonical LeRobot v2.1 dataset:

```text
dataset_root/
├── meta/
│   ├── info.json                    # cameras have dtype=video
│   ├── episodes.jsonl
│   └── lingbot_conversion.json
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet   # action/state/index bookkeeping
└── videos/
    └── chunk-000/
        └── observation.images.cam_high/
            └── episode_000000.mp4
```

The converter uses LeRobot's native `dtype=video` feature writer. Its metadata
uses CHW video shape semantics (`[3, H, W]`) and LeRobot owns video metadata,
MP4 creation, and temporary-frame cleanup.

Run `preprocess/extract_latents_from_pixels_adaptv1.py` on this layout. It reads
the MP4 files under `videos/` directly and applies deterministic frame-stride
sampling.

## 4. Extract Step 3 latents

Use `preprocess/extract_latents_from_pixels_adaptv1.py`. The extractor:

- reads camera MP4 files from `videos/`;
- writes per-camera, per-segment `.pth` files under `latents/`;
- samples frames according to `ori_fps` and target `fps`;
- saves latent and text-embedding tensors on CPU so the `.pth` files do not
  retain CUDA device information.

The extractor defaults to `--episodes-per-chunk 500`, while LeRobot v2.1 writes
1000 episodes per chunk by default. For datasets produced by this converter,
pass the chunk size explicitly:

```bash
--episodes-per-chunk 1000
```

More importantly, its `--fps` argument is not currently a reliable physical
sampling rate for arbitrary segment lengths. The restored script can be used to
verify legacy file I/O, but its latent `frame_ids` should **not** be considered
4DWAM training-ready merely because Step 3 completes.

### Required future Step 3 timing contract

The local RoboTwin / 4DWAM configuration has:

```text
action_per_frame = 16
Wan VAE temporal compression = 4
```

The loader computes action supervision from:

```text
actions per VAE latent frame = source_frame_stride * 4
```

Thus a valid LIFT2 extractor must enforce:

```text
source_frame_stride = 4
4 * 4 = 16 actions per VAE latent frame
```

For a 60 Hz LeRobot dataset this means the VAE input sampling rate is 15 Hz.
It should write deterministic frame IDs such as:

```text
0, 4, 8, 12, ...
```

and choose an input frame count compatible with the Wan temporal convention
(normally `4k + 1`). A future video-aware LIFT2 Step 3 implementation should:

1. read `meta/info.json` for the dataset fps and chunks size;
2. require `action_config` instead of fabricating a fallback segment;
3. read frames from MP4 or an explicit compatibility input;
4. use `--frame-stride 4`, not segment-length-derived sampling;
5. verify `frame_stride * 4 == action_per_frame` before writing a latent;
6. write its exact `frame_ids`, source fps, sampled fps, and latent dimensions.

Do not run 4DWAM training on latent artifacts until this timing contract is
implemented and validated.

## 5. Converter command line

Run from the LingBot-VA repository:

```bash
cd /soft/wangxi/4DWAM/lingbot-va
uv run python preprocess/convert_hdf5_to_lingbot_lerobot.py --help
```

### Direct command: restored Step 3 compatibility

```bash
uv run python preprocess/convert_hdf5_to_lingbot_lerobot.py \
  --data-dir /soft/wangxi/4DWAM/datasets/dataset_0402_wrench \
  --repo-id lift2_wrench_step3_compatible_60hz \
  --output-root /soft/wangxi/4DWAM/datasets_converted \
  --task-description "Open the toolbox, inspect its contents, and find the wrench." \
  --source-fps 60 \
  --fps 60 \
  --media-layout step3-compatible \
  --gripper-mode normalize \
  --overwrite
```

### Direct command: native LeRobot video storage

```bash
uv run python preprocess/convert_hdf5_to_lingbot_lerobot.py \
  --data-dir /soft/wangxi/4DWAM/datasets/dataset_0402_wrench \
  --repo-id lift2_wrench_standard_video_60hz \
  --output-root /soft/wangxi/4DWAM/datasets_converted \
  --task-description "Open the toolbox, inspect its contents, and find the wrench." \
  --source-fps 60 \
  --fps 60 \
  --media-layout standard-video \
  --gripper-mode normalize \
  --overwrite
```

`--overwrite` is deliberately required before an existing output root is
deleted. Use `--max-episodes 2` for a first conversion smoke test.

## 6. YAML configuration

The converter accepts the checked-in `preprocess/*conversion*.yaml` files. A
safe LIFT2 configuration looks like this:

```yaml
defaults:
  output_root: /soft/wangxi/4DWAM/datasets_converted
  source_fps: 60
  fps: 60
  media_layout: step3-compatible
  gripper_mode: normalize
  overwrite: false
  max_episodes: null

repo_id: lift2_wrench_step3_compatible_60hz

datasets:
  - data_dir: /soft/wangxi/4DWAM/datasets/dataset_0402_wrench
    task_description: Open the toolbox, inspect its contents, and find the wrench.
```

Legacy keys are intentionally rejected when enabled:

```yaml
skip_static_start: true
push_to_hub: true
```

Remove those keys or set them to `false`. The converter does not publish to the
Hub; publication must follow structural, temporal, and training validation.

Run a YAML conversion with:

```bash
uv run python preprocess/convert_hdf5_to_lingbot_lerobot.py \
  --config-path preprocess/my_lift2_conversion_single.yaml
```

## 7. Step 4 trace requirements

The current `extract_trace_from_ta.py` follows the latent tree:

```text
latents/chunk-000/<camera>/episode_000000_0_1582.pth
trace/chunk-000/<camera>/episode_000000_0_1582.pth
```

It reads `frame_ids` from each latent `.pth` and then reads the same image
indices from the matching MP4. This means trace is only trustworthy when Step
3's frame IDs are trustworthy.

Before accepting trace output, require all of the following:

```text
trace file name == latent file name
trace uses exactly the latent frame_ids
trace temporal dimension == latent_num_frames
trace token count == latent_height * latent_width
```

The current trace script only warns on one temporal mismatch. Treat that warning
as a failed preprocessing run, not a nonfatal condition.

## 8. Validation gates

### 8.1 Raw conversion gate

For every converted episode, check:

```text
action shape is [T, 14]
observation.state shape is [T, 14]
all three cameras exist and have T frames
all action_config segments satisfy 0 <= start < end <= T
meta/info.json fps is 60
```

For `step3-compatible` mode, also check all three sidecar MP4 files exist.
For `standard-video` mode, check all three metadata feature dtypes are `video`

### 8.2 Latent gate

For every `action_config` segment and camera, a `.pth` file must exist with:

```text
episode_index
start_frame
end_frame
frame_ids
latent
latent_num_frames
latent_height
latent_width
latent_channels
text_emb
text_emb_n
text_emb_d
```

The following must hold:

```text
latent.shape[0] == latent_num_frames * latent_height * latent_width
all adjacent frame_ids have the required fixed source stride
len(frame_ids) follows the selected Wan temporal convention
```

### 8.3 Loader and training gate

Before full preprocessing or FSDP, load two episodes and call:

```python
sample = dataset[0]
```

Verify:

```text
all three latent paths are discovered
the T-shape latent mosaic can be constructed
actions have 30 channels and 16 action slots per latent frame
trace can be rearranged and concatenated when enable_trace=True
no normalization divide-by-zero or shape assertion occurs
```

Then run a single-GPU, ten-step training smoke test. Only after that succeeds
should the full LIFT2 dataset be converted, latent-preprocessed, trace-
preprocessed, and launched with FSDP.

## 9. LIFT2 normalization

Do not reuse RobotWin's `q01` and `q99` directly. LIFT2 statistics must be
computed after the exact transformation used by the local loader:

```text
absolute 14D LIFT2 EEF action
  -> quaternion 16D action
  -> per-action_config relative pose
  -> 30D action-channel mapping
  -> q01 / q99 calculation
```

Computing quantiles directly from HDF5 `action_eef` produces statistics for a
different representation and is not valid for the current 4DWAM training path.

## 10. Recommended next implementation order

1. Convert two episodes in `step3-compatible` mode and verify restored Step 3
   and Step 4 file I/O only.
2. Add a separate video-aware LIFT2 latent extractor; do not modify the
   restored upstream Step 3 file.
3. Implement deterministic `--frame-stride 4` sampling and enforce the 16
   actions-per-latent relation.
4. Make trace extraction fail on latent/trace temporal or spatial mismatch.
5. Fix `LatentLeRobotDataset` paths to use `self.root / "latents"` and
   `self.root / "trace"`, and remove its `chunk-000` assumption.
6. Compute LIFT2 post-transform action quantiles.
7. Run the two-episode loader and ten-step training gates.

Only step 7 establishes that the complete LIFT2 -> 4DWAM chain is semantically
valid. A completed HDF5 conversion alone does not establish that fact.
