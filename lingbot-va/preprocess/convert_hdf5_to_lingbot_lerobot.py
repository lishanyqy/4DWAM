"""Convert LIFT2 HDF5 demonstrations into a LingBot-VA LeRobot dataset.

This converter intentionally has two explicit media layouts because the local
repository currently contains two incompatible requirements:

* ``step3-compatible`` (default) writes LeRobot ``dtype=image`` camera columns
  into parquet. After every episode has been saved, the converter also creates
  MP4 sidecars under ``videos/`` for
  ``extract_latents_from_pixels_adaptv1.py`` and ``extract_trace_from_ta.py``.
  This is a compatibility bridge, not a canonical video-only LeRobot dataset.
* ``standard-video`` writes native LeRobot ``dtype=video`` camera features and
  MP4 files. It is the canonical LeRobot layout, but it is deliberately
  incompatible with the restored parquet-image Step 3 script.

The converter never resamples time. For the current LIFT2 / 4DWAM setup, keep
the 60 Hz HDF5 timeline unchanged. Temporal sampling for Wan VAE must be done
later by a Step 3 implementation that explicitly enforces the relationship
between source frame stride, Wan temporal compression, and ``action_per_frame``.

The raw action contract is intentionally minimal:

* ``action``: absolute 14D EEF command from HDF5 ``action_eef``;
* ``observation.state``: 14D ``observations/qpos``;
* camera order: high, left wrist, right wrist;
* ``action_config``: explicit full-episode task segment.

The LingBot loader owns the later 14D Euler -> 16D quaternion conversion,
segment-relative pose calculation, 30D channel alignment, and normalization.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import cv2
import h5py
import numpy as np
import yaml

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import encode_video_frames
except ImportError:
    # OpenPI environments pin an older LeRobot import path.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.video_utils import encode_video_frames


DEFAULT_OUTPUT_ROOT = Path("/soft/wangxi/4DWAM/datasets_converted")
DEFAULT_FPS = 60
LEROBOT_CHUNK_SIZE = 500
RAW_GRIPPER_MIN = 0.0
RAW_GRIPPER_MAX = 5.0

MEDIA_LAYOUT_STEP3_COMPATIBLE = "step3-compatible"
MEDIA_LAYOUT_STANDARD_VIDEO = "standard-video"
MediaLayout = Literal["step3-compatible", "standard-video"]
GripperMode = Literal["normalize", "raw"]

ACTION_EEF_KEY = "action_eef"
STATE_QPOS_KEY = "observations/qpos"


@dataclass(frozen=True)
class CameraSpec:
    """Describe one required source camera and its LingBot feature key."""

    name: str
    hdf5_key: str
    feature_key: str


CAMERA_SPECS = (
    CameraSpec("high", "observations/images/head", "observation.images.cam_high"),
    CameraSpec("left_wrist", "observations/images/left_wrist", "observation.images.cam_left_wrist"),
    CameraSpec("right_wrist", "observations/images/right_wrist", "observation.images.cam_right_wrist"),
)


@dataclass(frozen=True)
class SourceDataset:
    """One directory of episode_*.hdf5 files and its task text."""

    data_dir: Path
    task_description: str


@dataclass(frozen=True)
class ConversionRequest:
    """Fully resolved conversion settings for one output dataset."""

    sources: tuple[SourceDataset, ...]
    repo_id: str
    output_root: Path
    source_fps: int
    fps: int
    media_layout: MediaLayout
    gripper_mode: GripperMode
    overwrite: bool
    max_episodes: int | None


def normalize_gripper(raw_values: np.ndarray | float) -> np.ndarray | float:
    """Map LIFT2 raw gripper values from [0, 5] into [0, 1]."""
    return np.clip(raw_values, RAW_GRIPPER_MIN, RAW_GRIPPER_MAX) / RAW_GRIPPER_MAX


def build_action(action_eef: np.ndarray, gripper_mode: GripperMode) -> np.ndarray:
    """Validate and return the raw 14D action used by the LingBot loader."""
    action = np.asarray(action_eef, dtype=np.float32).copy()
    if action.shape != (14,):
        raise ValueError(f"Expected action_eef frame shape (14,), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("action_eef contains NaN or Inf")

    if gripper_mode == "normalize":
        action[6] = normalize_gripper(action[6])
        action[13] = normalize_gripper(action[13])
    elif gripper_mode != "raw":
        raise ValueError(f"Unsupported gripper mode: {gripper_mode}")
    return action


def build_state(qpos: np.ndarray) -> np.ndarray:
    """Validate and return the 14D proprioceptive qpos observation."""
    state = np.asarray(qpos, dtype=np.float32).copy()
    if state.shape != (14,):
        raise ValueError(f"Expected observations/qpos frame shape (14,), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("observations/qpos contains NaN or Inf")
    return state


def decode_rgb_image(encoded_image: np.ndarray, dataset_key: str, frame_index: int) -> np.ndarray:
    """Decode one JPEG/PNG HDF5 byte buffer as an HWC RGB image."""
    decoded_bgr = cv2.imdecode(np.asarray(encoded_image), cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise ValueError(f"Failed to decode {dataset_key} frame {frame_index}")
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def parse_episode_number(episode_path: Path) -> int:
    """Read the numeric suffix from ``episode_123.hdf5``."""
    match = re.fullmatch(r"episode_(\d+)", episode_path.stem)
    if match is None:
        raise ValueError(f"Episode file must be named episode_<integer>.hdf5: {episode_path}")
    return int(match.group(1))


def list_episode_files(data_dir: Path) -> list[Path]:
    """Return numerically ordered HDF5 episode files for a source directory."""
    if not data_dir.is_dir():
        raise NotADirectoryError(f"HDF5 source directory does not exist: {data_dir}")
    episode_files = sorted(data_dir.glob("episode_*.hdf5"), key=parse_episode_number)
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {data_dir}")
    return episode_files


def validate_hdf5_episode(hdf5_file: h5py.File, episode_file: Path) -> int:
    """Validate the required LIFT2 fields and return the episode length."""
    required_keys = [ACTION_EEF_KEY, STATE_QPOS_KEY, *(spec.hdf5_key for spec in CAMERA_SPECS)]
    missing_keys = [key for key in required_keys if key not in hdf5_file]
    if missing_keys:
        raise KeyError(f"{episode_file} is missing required fields: {missing_keys}")

    action_dataset = hdf5_file[ACTION_EEF_KEY]
    state_dataset = hdf5_file[STATE_QPOS_KEY]
    if action_dataset.ndim != 2 or action_dataset.shape[1] != 14:
        raise ValueError(f"{episode_file}:{ACTION_EEF_KEY} must have shape [T, 14], got {action_dataset.shape}")
    if state_dataset.ndim != 2 or state_dataset.shape[1] != 14:
        raise ValueError(f"{episode_file}:{STATE_QPOS_KEY} must have shape [T, 14], got {state_dataset.shape}")

    episode_length = int(action_dataset.shape[0])
    if episode_length <= 0:
        raise ValueError(f"{episode_file}:{ACTION_EEF_KEY} is empty")
    if state_dataset.shape[0] != episode_length:
        raise ValueError(f"{episode_file}: qpos length differs from action_eef length")

    for camera_spec in CAMERA_SPECS:
        camera_dataset = hdf5_file[camera_spec.hdf5_key]
        if camera_dataset.ndim != 2 or camera_dataset.shape[0] != episode_length:
            raise ValueError(
                f"{episode_file}:{camera_spec.hdf5_key} must have shape [T, N] with T={episode_length}, "
                f"got {camera_dataset.shape}"
            )
    return episode_length


def infer_camera_shapes(first_episode_file: Path) -> dict[str, tuple[int, int, int]]:
    """Read the first frame of each view to determine expected HWC dimensions."""
    with h5py.File(first_episode_file, "r") as hdf5_file:
        validate_hdf5_episode(hdf5_file, first_episode_file)
        shapes: dict[str, tuple[int, int, int]] = {}
        for camera_spec in CAMERA_SPECS:
            image = decode_rgb_image(hdf5_file[camera_spec.hdf5_key][0], camera_spec.hdf5_key, 0)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"{first_episode_file}:{camera_spec.hdf5_key} must decode to HWC RGB, got {image.shape}"
                )
            shapes[camera_spec.feature_key] = tuple(int(value) for value in image.shape)
    return shapes


def create_features(
    camera_shapes_hwc: dict[str, tuple[int, int, int]],
    media_layout: MediaLayout,
) -> dict[str, dict[str, Any]]:
    """Create LeRobot features for the selected explicit media layout."""
    features: dict[str, dict[str, Any]] = {}
    for camera_spec in CAMERA_SPECS:
        image_height, image_width, image_channels = camera_shapes_hwc[camera_spec.feature_key]
        if media_layout == MEDIA_LAYOUT_STEP3_COMPATIBLE:
            features[camera_spec.feature_key] = {
                "dtype": "image",
                "shape": (image_height, image_width, image_channels),
                "names": ["height", "width", "channel"],
            }
        else:
            features[camera_spec.feature_key] = {
                "dtype": "video",
                "shape": (image_channels, image_height, image_width),
                "names": ["rgb", "height", "width"],
            }

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (14,),
        "names": ["joint_state"],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (14,),
        "names": ["absolute_eef_action"],
    }
    return features


def create_dataset(request: ConversionRequest, camera_shapes_hwc: dict[str, tuple[int, int, int]]) -> LeRobotDataset:
    """Create a LeRobot writer with the requested storage contract."""
    output_path = request.output_root / request.repo_id
    uses_native_videos = request.media_layout == MEDIA_LAYOUT_STANDARD_VIDEO
    dataset = LeRobotDataset.create(
        repo_id=request.repo_id,
        root=output_path,
        robot_type="bimanual",
        fps=request.fps,
        features=create_features(camera_shapes_hwc, request.media_layout),
        use_videos=uses_native_videos,
        image_writer_processes=5,
        image_writer_threads=10,
        batch_encoding_size=1,
    )

    # This LeRobot version does not expose chunk size through create(). Its
    # writer reads the value from metadata for every episode, so set it before
    # the first save_episode() call to keep data, videos, latents, and traces on
    # the same 500-episode chunk boundary.
    dataset.meta.info["chunks_size"] = LEROBOT_CHUNK_SIZE
    if dataset.meta.chunks_size != LEROBOT_CHUNK_SIZE:
        raise RuntimeError(
            "Failed to configure LeRobot chunk size: expected "
            f"{LEROBOT_CHUNK_SIZE}, got {dataset.meta.chunks_size}"
        )
    return dataset


def build_frame(
    hdf5_file: h5py.File,
    frame_index: int,
    action_eef: np.ndarray,
    qpos: np.ndarray,
    camera_shapes_hwc: dict[str, tuple[int, int, int]],
    gripper_mode: GripperMode,
) -> dict[str, np.ndarray]:
    """Build one synchronized LeRobot frame without resampling or interpolation."""
    frame: dict[str, np.ndarray] = {
        "action": build_action(action_eef[frame_index], gripper_mode),
        "observation.state": build_state(qpos[frame_index]),
    }
    for camera_spec in CAMERA_SPECS:
        image = decode_rgb_image(
            hdf5_file[camera_spec.hdf5_key][frame_index],
            camera_spec.hdf5_key,
            frame_index,
        )
        expected_shape = camera_shapes_hwc[camera_spec.feature_key]
        if image.shape != expected_shape:
            raise ValueError(
                f"{camera_spec.hdf5_key} frame {frame_index} shape changed from {expected_shape} to {image.shape}"
            )
        frame[camera_spec.feature_key] = image
    return frame


def get_video_path(dataset_root: Path, episode_index: int, camera_key: str) -> Path:
    """Return the standard v2.1 MP4 location for an episode and camera."""
    chunk_index = episode_index // LEROBOT_CHUNK_SIZE
    return (
        dataset_root
        / "videos"
        / f"chunk-{chunk_index:03d}"
        / camera_key
        / f"episode_{episode_index:06d}.mp4"
    )


def get_image_episode_dir(dataset_root: Path, episode_index: int, camera_key: str) -> Path:
    """Return LeRobot's on-disk image directory for an image-backed episode."""
    return dataset_root / "images" / camera_key / f"episode_{episode_index:06d}"


def encode_step3_compatible_sidecar_videos(dataset_root: Path, episode_indices: list[int], fps: int) -> None:
    """Encode trace sidecars after image-backed recording has completely finished.

    This must run after all ``save_episode`` calls. A LeRobot image dataset has
    no native video keys, so adding MP4 files while it is still recording causes
    its internal episode checks to reject the unexpected files.
    """
    for episode_index in episode_indices:
        for camera_spec in CAMERA_SPECS:
            image_directory = get_image_episode_dir(dataset_root, episode_index, camera_spec.feature_key)
            if not image_directory.is_dir():
                raise FileNotFoundError(
                    "Could not create Step 4 video sidecar because the LeRobot image directory is missing: "
                    f"{image_directory}"
                )
            video_path = get_video_path(dataset_root, episode_index, camera_spec.feature_key)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            encode_video_frames(image_directory, video_path, fps=fps, overwrite=True)


def patch_episode_action_configs(
    dataset_root: Path,
    expected_lengths: dict[int, int],
    task_descriptions: dict[int, str],
) -> None:
    """Add explicit full-episode LingBot action_config metadata to every row."""
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"LeRobot did not create episode metadata: {episodes_path}")

    patched_lines: list[str] = []
    found_episode_indices: set[int] = set()
    with episodes_path.open("r", encoding="utf-8") as episode_file:
        for line_number, raw_line in enumerate(episode_file, start=1):
            if not raw_line.strip():
                continue
            metadata = json.loads(raw_line)
            episode_index = int(metadata["episode_index"])
            if episode_index not in expected_lengths:
                raise ValueError(f"Unexpected episode_index={episode_index} in {episodes_path}:{line_number}")

            episode_length = int(metadata.get("length", metadata.get("num_frames", -1)))
            if episode_length != expected_lengths[episode_index]:
                raise ValueError(
                    f"Episode {episode_index} metadata length={episode_length} does not match "
                    f"the {expected_lengths[episode_index]} frames written by the converter"
                )

            task_description = task_descriptions[episode_index]
            metadata["tasks"] = [task_description]
            metadata["action_config"] = [
                {
                    "start_frame": 0,
                    "end_frame": episode_length,
                    "action_text": task_description,
                    "skill": "",
                }
            ]
            patched_lines.append(json.dumps(metadata, ensure_ascii=True) + "\n")
            found_episode_indices.add(episode_index)

    missing_episode_indices = set(expected_lengths) - found_episode_indices
    if missing_episode_indices:
        raise ValueError(f"LeRobot metadata is missing converted episodes: {sorted(missing_episode_indices)}")

    with episodes_path.open("w", encoding="utf-8") as episode_file:
        episode_file.writelines(patched_lines)


def write_conversion_manifest(request: ConversionRequest, saved_episode_count: int) -> None:
    """Write converter-specific facts without corrupting LeRobot's own metadata."""
    output_path = request.output_root / request.repo_id
    manifest = {
        "converter": "convert_hdf5_to_lingbot_lerobot.py",
        "source_fps": request.source_fps,
        "dataset_fps": request.fps,
        "media_layout": request.media_layout,
        "gripper_mode": request.gripper_mode,
        "episode_count": saved_episode_count,
        "resampling": "disabled",
        "static_prefix_trimming": "disabled",
        "camera_order": [camera_spec.feature_key for camera_spec in CAMERA_SPECS],
        "action_source": ACTION_EEF_KEY,
        "state_source": STATE_QPOS_KEY,
    }
    manifest_path = output_path / "meta" / "lingbot_conversion.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")


def verify_output(request: ConversionRequest, saved_episode_indices: list[int]) -> None:
    """Perform structural checks that do not require loading model weights."""
    output_path = request.output_root / request.repo_id
    info_path = output_path / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as info_file:
        info = json.load(info_file)

    expected_dtype = "image" if request.media_layout == MEDIA_LAYOUT_STEP3_COMPATIBLE else "video"
    for camera_spec in CAMERA_SPECS:
        actual_dtype = info["features"][camera_spec.feature_key]["dtype"]
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"{camera_spec.feature_key} has dtype={actual_dtype!r}; expected {expected_dtype!r}"
            )

    if request.media_layout == MEDIA_LAYOUT_STEP3_COMPATIBLE:
        for episode_index in saved_episode_indices:
            for camera_spec in CAMERA_SPECS:
                video_path = get_video_path(output_path, episode_index, camera_spec.feature_key)
                if not video_path.is_file():
                    raise FileNotFoundError(f"Missing Step 4 sidecar video: {video_path}")
    elif (output_path / "images").exists():
        raise ValueError("Native LeRobot video mode left a permanent images/ directory behind")


def convert(request: ConversionRequest) -> None:
    """Write one output dataset from one or more homogeneous LIFT2 sources."""
    if request.source_fps != request.fps:
        raise ValueError(
            "This converter intentionally does not resample. source_fps and fps must be equal; "
            f"got source_fps={request.source_fps}, fps={request.fps}."
        )
    if request.fps <= 0:
        raise ValueError(f"fps must be positive, got {request.fps}")
    if not request.sources:
        raise ValueError("At least one source dataset is required")

    source_episode_files = [(source, list_episode_files(source.data_dir)) for source in request.sources]
    first_episode_file = source_episode_files[0][1][0]
    camera_shapes_hwc = infer_camera_shapes(first_episode_file)

    output_path = request.output_root / request.repo_id
    request.output_root.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not request.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Pass --overwrite only after confirming it can be removed."
            )
        shutil.rmtree(output_path)

    print(f"Writing {request.media_layout} dataset to {output_path}")
    print(f"Keeping the original {request.fps} Hz timeline; no trim or resampling will occur.")
    dataset = create_dataset(request, camera_shapes_hwc)

    saved_episode_indices: list[int] = []
    expected_lengths: dict[int, int] = {}
    task_descriptions: dict[int, str] = {}
    saved_episode_count = 0

    for source, episode_files in source_episode_files:
        for episode_file in episode_files:
            if request.max_episodes is not None and saved_episode_count >= request.max_episodes:
                break

            print(f"Converting {episode_file.name} as output episode {saved_episode_count}")
            with h5py.File(episode_file, "r") as hdf5_file:
                episode_length = validate_hdf5_episode(hdf5_file, episode_file)
                action_eef = np.asarray(hdf5_file[ACTION_EEF_KEY][:], dtype=np.float32)
                qpos = np.asarray(hdf5_file[STATE_QPOS_KEY][:], dtype=np.float32)
                if not np.isfinite(action_eef).all() or not np.isfinite(qpos).all():
                    raise ValueError(f"{episode_file} contains NaN or Inf in action_eef or qpos")

                for frame_index in range(episode_length):
                    dataset.add_frame(
                        build_frame(
                            hdf5_file=hdf5_file,
                            frame_index=frame_index,
                            action_eef=action_eef,
                            qpos=qpos,
                            camera_shapes_hwc=camera_shapes_hwc,
                            gripper_mode=request.gripper_mode,
                        ),
                        task=source.task_description,
                    )
                dataset.save_episode()

            expected_lengths[saved_episode_count] = episode_length
            task_descriptions[saved_episode_count] = source.task_description
            saved_episode_indices.append(saved_episode_count)
            saved_episode_count += 1

        if request.max_episodes is not None and saved_episode_count >= request.max_episodes:
            break

    if saved_episode_count == 0:
        raise RuntimeError("No episodes were converted")

    patch_episode_action_configs(output_path, expected_lengths, task_descriptions)
    if request.media_layout == MEDIA_LAYOUT_STEP3_COMPATIBLE:
        encode_step3_compatible_sidecar_videos(output_path, saved_episode_indices, request.fps)
    write_conversion_manifest(request, saved_episode_count)
    verify_output(request, saved_episode_indices)

    print(f"Converted {saved_episode_count} episodes successfully.")
    print(f"Dataset root: {output_path}")
    if request.media_layout == MEDIA_LAYOUT_STEP3_COMPATIBLE:
        print(
            "This output is intentionally image-backed for the restored Step 3 script. "
            "Run Step 3 with --episodes-per-chunk 1000, then run Step 4 against the generated videos/."
        )
    else:
        print(
            "This is native LeRobot video storage. The restored parquet-image Step 3 script cannot consume it."
        )


def get_config_value(mapping: dict[str, Any], defaults: dict[str, Any], key: str, fallback: Any = None) -> Any:
    """Resolve a config key with item -> defaults -> fallback precedence."""
    if key in mapping and mapping[key] is not None:
        return mapping[key]
    if key in defaults and defaults[key] is not None:
        return defaults[key]
    return fallback


def build_request_from_yaml(config_path: Path) -> ConversionRequest:
    """Parse the documented single- or multi-source conversion YAML format."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"YAML root must be a mapping: {config_path}")

    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("YAML defaults must be a mapping")
    source_configs = config.get("datasets")
    if source_configs is None:
        source_configs = [{key: value for key, value in config.items() if key != "defaults"}]
    if not isinstance(source_configs, list) or not source_configs:
        raise ValueError("YAML datasets must be a non-empty list")

    sources: list[SourceDataset] = []
    for source_config in source_configs:
        if not isinstance(source_config, dict):
            raise ValueError("Every YAML datasets item must be a mapping")
        data_dir = get_config_value(source_config, defaults, "data_dir")
        task_description = get_config_value(source_config, defaults, "task_description", config.get("task_description"))
        if not data_dir or not task_description:
            raise ValueError("Every source must define data_dir and task_description")
        sources.append(SourceDataset(Path(str(data_dir)), str(task_description)))

    if get_config_value(config, defaults, "skip_static_start", False):
        raise ValueError(
            "skip_static_start is intentionally unsupported. Remove it or set it to false; "
            "this converter preserves the original episode timeline."
        )
    if get_config_value(config, defaults, "push_to_hub", False):
        raise ValueError("push_to_hub is intentionally unsupported by this converter; publish only after validation.")

    repo_id = get_config_value(config, defaults, "repo_id")
    if not repo_id:
        raise ValueError("YAML must define repo_id")
    return ConversionRequest(
        sources=tuple(sources),
        repo_id=str(repo_id),
        output_root=Path(str(get_config_value(config, defaults, "output_root", DEFAULT_OUTPUT_ROOT))),
        source_fps=int(get_config_value(config, defaults, "source_fps", DEFAULT_FPS)),
        fps=int(get_config_value(config, defaults, "fps", DEFAULT_FPS)),
        media_layout=str(get_config_value(config, defaults, "media_layout", MEDIA_LAYOUT_STEP3_COMPATIBLE)),
        gripper_mode=str(get_config_value(config, defaults, "gripper_mode", "normalize")),
        overwrite=bool(get_config_value(config, defaults, "overwrite", False)),
        max_episodes=get_config_value(config, defaults, "max_episodes", None),
    )


def build_request_from_arguments(arguments: argparse.Namespace) -> ConversionRequest:
    """Build a direct-command request when a YAML configuration is not used."""
    if not arguments.data_dir or not arguments.repo_id or not arguments.task_description:
        raise ValueError("Direct mode requires --data-dir, --repo-id, and --task-description")
    return ConversionRequest(
        sources=(SourceDataset(Path(arguments.data_dir), arguments.task_description),),
        repo_id=arguments.repo_id,
        output_root=Path(arguments.output_root),
        source_fps=arguments.source_fps,
        fps=arguments.fps,
        media_layout=arguments.media_layout,
        gripper_mode=arguments.gripper_mode,
        overwrite=arguments.overwrite,
        max_episodes=arguments.max_episodes,
    )


def parse_arguments() -> argparse.Namespace:
    """Define the intentionally narrow no-resampling conversion CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, help="YAML configuration for a single or merged dataset")
    parser.add_argument("--data-dir", help="Directory containing episode_*.hdf5 files in direct mode")
    parser.add_argument("--repo-id", help="Output dataset directory name in direct mode")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Parent directory for output datasets")
    parser.add_argument("--task-description", help="Full-episode LingBot action text in direct mode")
    parser.add_argument("--source-fps", type=int, default=DEFAULT_FPS, help="Recorded HDF5 rate; must equal --fps")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="LeRobot rate; must equal --source-fps")
    parser.add_argument(
        "--media-layout",
        choices=[MEDIA_LAYOUT_STEP3_COMPATIBLE, MEDIA_LAYOUT_STANDARD_VIDEO],
        default=MEDIA_LAYOUT_STEP3_COMPATIBLE,
        help="Explicit storage contract; use step3-compatible while the original Step 3 is restored",
    )
    parser.add_argument("--gripper-mode", choices=["normalize", "raw"], default="normalize")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing same-name output dataset")
    parser.add_argument("--max-episodes", type=int, help="Convert only this many episodes for a smoke test")
    return parser.parse_args()


def main() -> None:
    """Run the converter from direct arguments or the documented YAML format."""
    arguments = parse_arguments()
    if arguments.config_path:
        request = build_request_from_yaml(arguments.config_path)
        if arguments.overwrite:
            request = replace(request, overwrite=True)
        if arguments.max_episodes is not None:
            request = replace(request, max_episodes=arguments.max_episodes)
    else:
        request = build_request_from_arguments(arguments)

    if request.media_layout not in {MEDIA_LAYOUT_STEP3_COMPATIBLE, MEDIA_LAYOUT_STANDARD_VIDEO}:
        raise ValueError(f"Unsupported media_layout: {request.media_layout}")
    if request.gripper_mode not in {"normalize", "raw"}:
        raise ValueError(f"Unsupported gripper_mode: {request.gripper_mode}")
    if request.max_episodes is not None and request.max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    convert(request)


if __name__ == "__main__":
    main()
