"""Compute action quantiles using the LingBot-VA training representation.

This script is intentionally separate from dataset conversion because the raw
LeRobot action is not the representation normalized by the model. It reproduces
the active action preprocessing in ``LatentLeRobotDataset``:

1. Load the stored 14D bimanual XYZ/Euler/gripper action.
2. Convert each Euler rotation to an XYZW quaternion, producing 16 channels.
3. Optionally re-anchor each trajectory segment to its first EEF pose.
4. Apply the same latent-frame alignment and leading action padding as training.
5. Map the 16 active channels into the canonical 30D model action layout.
6. Compute per-channel q01 and q99 values.

The script only reads the converted dataset and extracted latent metadata. It
writes one JSON statistics file and never modifies parquet, latent, or trace
artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm


PoseMode = Literal["segment-relative", "stored"]

DEFAULT_DATASET_ROOT = Path(
    "/soft/wangxi/4DWAM/datasets_converted/"
    "lift2_merged_step3_compatible_60hz"
)
DEFAULT_CAMERA_KEY = "observation.images.cam_high"
DEFAULT_USED_ACTION_CHANNEL_IDS = (
    list(range(0, 7))
    + list(range(28, 29))
    + list(range(7, 14))
    + list(range(29, 30))
)
MODEL_ACTION_DIMENSION = 30
EEF_EULER_ACTION_DIMENSION = 14
EEF_QUATERNION_ACTION_DIMENSION = 16
LEFT_GRIPPER_MODEL_CHANNEL = 28
RIGHT_GRIPPER_MODEL_CHANNEL = 29


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute q01/q99 over actions after applying the same representation "
            "and temporal alignment used by LingBot-VA training."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Converted LeRobot dataset containing meta, data, and latents.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "<dataset-root>/meta/action_norm_stats.json."
        ),
    )
    parser.add_argument(
        "--pose-mode",
        choices=("segment-relative", "stored"),
        default="segment-relative",
        help=(
            "Use 'segment-relative' to match the current robotwin_tshape loader, "
            "or 'stored' to preserve the EEF reference already stored on disk."
        ),
    )
    parser.add_argument(
        "--camera-key",
        default=DEFAULT_CAMERA_KEY,
        help="Camera latent metadata used to obtain frame_ids.",
    )
    parser.add_argument(
        "--lower-quantile",
        type=float,
        default=0.01,
        help="Lower action quantile. Default: 0.01.",
    )
    parser.add_argument(
        "--upper-quantile",
        type=float,
        default=0.99,
        help="Upper action quantile. Default: 0.99.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Optional segment limit for a quick validation run.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if not arguments.dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset root does not exist: {arguments.dataset_root}"
        )
    if not 0.0 <= arguments.lower_quantile < arguments.upper_quantile <= 1.0:
        raise ValueError(
            "Quantiles must satisfy 0 <= lower < upper <= 1, got "
            f"{arguments.lower_quantile} and {arguments.upper_quantile}."
        )
    if arguments.max_segments is not None and arguments.max_segments <= 0:
        raise ValueError("--max-segments must be positive when provided.")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                records.append(json.loads(stripped_line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from error
    return records


def resolve_episode_chunk(episode_index: int, chunk_size: int) -> int:
    if chunk_size <= 0:
        raise ValueError(f"Invalid dataset chunk size: {chunk_size}")
    return episode_index // chunk_size


def resolve_episode_parquet_path(
    dataset_root: Path,
    data_path_template: str,
    episode_index: int,
    episode_chunk: int,
) -> Path:
    relative_path = data_path_template.format(
        episode_index=episode_index,
        episode_chunk=episode_chunk,
    )
    return dataset_root / relative_path


def load_episode_actions(parquet_path: Path) -> np.ndarray:
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Episode parquet does not exist: {parquet_path}")

    action_series = pd.read_parquet(parquet_path, columns=["action"])["action"]
    action_rows = [np.asarray(action_row, dtype=np.float64) for action_row in action_series]
    actions = np.stack(action_rows, axis=0)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(
            f"Expected action shape [T, 14] in {parquet_path}, got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError(f"Action contains NaN or Inf: {parquet_path}")
    return actions


def load_latent_frame_ids(
    dataset_root: Path,
    camera_key: str,
    episode_index: int,
    episode_chunk: int,
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    latent_path = (
        dataset_root
        / "latents"
        / f"chunk-{episode_chunk:03d}"
        / camera_key
        / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
    )
    if not latent_path.is_file():
        raise FileNotFoundError(f"Latent metadata does not exist: {latent_path}")

    latent_record = torch.load(latent_path, map_location="cpu", weights_only=False)
    if "frame_ids" not in latent_record:
        raise KeyError(f"Latent record has no frame_ids: {latent_path}")

    frame_ids = np.asarray(latent_record["frame_ids"], dtype=np.int64)
    if frame_ids.ndim != 1 or frame_ids.size < 2:
        raise ValueError(
            f"Expected at least two one-dimensional frame_ids in {latent_path}, "
            f"got shape {frame_ids.shape}"
        )
    frame_differences = np.diff(frame_ids)
    if np.any(frame_differences <= 0):
        raise ValueError(f"frame_ids are not strictly increasing: {latent_path}")
    if not np.all(frame_differences == frame_differences[0]):
        raise ValueError(f"frame_ids do not have a constant stride: {latent_path}")
    return frame_ids


def convert_xyz_euler_actions_to_quaternions(actions: np.ndarray) -> np.ndarray:
    """Match LatentLeRobotDataset's 14D Euler to 16D XYZW conversion."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != EEF_EULER_ACTION_DIMENSION:
        raise ValueError(
            "Expected EEF Euler actions with shape [T, 14], got "
            f"{actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("EEF Euler actions contain NaN or Inf values.")

    left_quaternions_xyzw = Rotation.from_euler(
        "xyz", actions[:, 3:6]
    ).as_quat()
    right_quaternions_xyzw = Rotation.from_euler(
        "xyz", actions[:, 10:13]
    ).as_quat()

    converted_actions = np.concatenate(
        [
            actions[:, :3],
            left_quaternions_xyzw,
            actions[:, 6:7],
            actions[:, 7:10],
            right_quaternions_xyzw,
            actions[:, 13:14],
        ],
        axis=1,
    )
    if converted_actions.shape[1] != EEF_QUATERNION_ACTION_DIMENSION:
        raise AssertionError(
            "Expected converted action width 16, got "
            f"{converted_actions.shape}"
        )

    quaternion_norms = np.concatenate(
        [
            np.linalg.norm(converted_actions[:, 3:7], axis=1),
            np.linalg.norm(converted_actions[:, 11:15], axis=1),
        ]
    )
    if not np.allclose(quaternion_norms, 1.0, rtol=1e-7, atol=1e-7):
        raise AssertionError(
            "Euler-to-quaternion conversion produced non-unit quaternions: "
            f"min_norm={quaternion_norms.min()}, "
            f"max_norm={quaternion_norms.max()}"
        )
    return converted_actions


def convert_pose_to_segment_relative(pose: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_quat(pose[:, 3:7])
    first_rotations = Rotation.from_quat(
        np.repeat(pose[:1, 3:7], repeats=pose.shape[0], axis=0)
    )
    relative_translation = pose[:, :3] - pose[:1, :3]
    relative_quaternion = (first_rotations.inv() * rotations).as_quat()
    return np.concatenate([relative_translation, relative_quaternion], axis=1)


def build_inverse_action_channel_ids() -> np.ndarray:
    active_action_width = len(DEFAULT_USED_ACTION_CHANNEL_IDS)
    if active_action_width != EEF_QUATERNION_ACTION_DIMENSION:
        raise AssertionError(
            "The canonical mapping must contain 16 active action channels, got "
            f"{active_action_width}"
        )
    if DEFAULT_USED_ACTION_CHANNEL_IDS[7] != LEFT_GRIPPER_MODEL_CHANNEL:
        raise AssertionError("16D action channel 7 must map to left gripper channel 28.")
    if DEFAULT_USED_ACTION_CHANNEL_IDS[15] != RIGHT_GRIPPER_MODEL_CHANNEL:
        raise AssertionError("16D action channel 15 must map to right gripper channel 29.")
    if len(set(DEFAULT_USED_ACTION_CHANNEL_IDS)) != active_action_width:
        raise AssertionError("Active model action channel IDs must be unique.")

    inverse_channel_ids = np.full(
        MODEL_ACTION_DIMENSION,
        active_action_width,
        dtype=np.int64,
    )
    for active_channel_index, model_channel_index in enumerate(
        DEFAULT_USED_ACTION_CHANNEL_IDS
    ):
        inverse_channel_ids[model_channel_index] = active_channel_index
    return inverse_channel_ids


def process_segment_actions(
    episode_actions: np.ndarray,
    start_frame: int,
    end_frame: int,
    latent_frame_ids: np.ndarray,
    pose_mode: PoseMode,
    inverse_action_channel_ids: np.ndarray,
) -> np.ndarray:
    segment_actions = episode_actions[start_frame:end_frame]
    action_shift = int(latent_frame_ids[0] - start_frame)
    if action_shift < 0 or action_shift >= segment_actions.shape[0]:
        raise ValueError(
            f"Invalid action shift {action_shift} for segment "
            f"[{start_frame}, {end_frame})"
        )

    aligned_actions = segment_actions[action_shift:]
    aligned_actions = convert_xyz_euler_actions_to_quaternions(aligned_actions)

    if pose_mode == "segment-relative":
        left_pose = convert_pose_to_segment_relative(aligned_actions[:, :7])
        right_pose = convert_pose_to_segment_relative(aligned_actions[:, 8:15])
        aligned_actions = np.concatenate(
            [
                left_pose,
                aligned_actions[:, 7:8],
                right_pose,
                aligned_actions[:, 15:16],
            ],
            axis=1,
        )

    frame_stride = int(latent_frame_ids[1] - latent_frame_ids[0])
    leading_padding_steps = frame_stride * 4
    aligned_actions = np.pad(
        aligned_actions,
        pad_width=((leading_padding_steps, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    latent_frame_count = (len(latent_frame_ids) - 1) // 4 + 1
    required_action_count = latent_frame_count * frame_stride * 4
    if aligned_actions.shape[0] < required_action_count:
        raise ValueError(
            f"Aligned action length {aligned_actions.shape[0]} is shorter than "
            f"required length {required_action_count} for segment "
            f"[{start_frame}, {end_frame})"
        )
    aligned_actions = aligned_actions[:required_action_count]

    actions_with_dummy_channel = np.pad(
        aligned_actions,
        pad_width=((0, 0), (0, 1)),
        mode="constant",
        constant_values=0,
    )
    model_actions = actions_with_dummy_channel[:, inverse_action_channel_ids]
    if model_actions.shape[1] != MODEL_ACTION_DIMENSION:
        raise AssertionError(
            "Expected canonical action width 30, got "
            f"{model_actions.shape}"
        )
    if not np.array_equal(model_actions[:, 28], aligned_actions[:, 7]):
        raise AssertionError("Left gripper was not mapped to model channel 28.")
    if not np.array_equal(model_actions[:, 29], aligned_actions[:, 15]):
        raise AssertionError("Right gripper was not mapped to model channel 29.")

    unused_model_channels = sorted(
        set(range(MODEL_ACTION_DIMENSION)) - set(DEFAULT_USED_ACTION_CHANNEL_IDS)
    )
    if np.any(model_actions[:, unused_model_channels] != 0.0):
        raise AssertionError("Unused model action channels must remain zero.")
    return model_actions


def collect_processed_actions(
    dataset_root: Path,
    camera_key: str,
    pose_mode: PoseMode,
    max_segments: int | None,
) -> tuple[np.ndarray, dict[str, int]]:
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    dataset_info = load_json(info_path)
    episode_records = load_json_lines(episodes_path)

    chunk_size = int(dataset_info["chunks_size"])
    data_path_template = str(dataset_info["data_path"])
    inverse_action_channel_ids = build_inverse_action_channel_ids()

    processed_segments: list[np.ndarray] = []
    episode_action_cache: dict[int, np.ndarray] = {}
    segment_count = 0

    total_segment_count = sum(
        len(episode_record.get("action_config", []))
        for episode_record in episode_records
    )
    if max_segments is not None:
        total_segment_count = min(total_segment_count, max_segments)

    progress_bar = tqdm(total=total_segment_count, desc="Processing action segments")
    try:
        for episode_record in episode_records:
            episode_index = int(episode_record["episode_index"])
            episode_chunk = resolve_episode_chunk(episode_index, chunk_size)

            if episode_index not in episode_action_cache:
                parquet_path = resolve_episode_parquet_path(
                    dataset_root,
                    data_path_template,
                    episode_index,
                    episode_chunk,
                )
                episode_action_cache[episode_index] = load_episode_actions(parquet_path)

            episode_actions = episode_action_cache[episode_index]
            for action_config in episode_record.get("action_config", []):
                if max_segments is not None and segment_count >= max_segments:
                    break

                start_frame = int(action_config["start_frame"])
                end_frame = int(action_config["end_frame"])
                latent_frame_ids = load_latent_frame_ids(
                    dataset_root=dataset_root,
                    camera_key=camera_key,
                    episode_index=episode_index,
                    episode_chunk=episode_chunk,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
                processed_actions = process_segment_actions(
                    episode_actions=episode_actions,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    latent_frame_ids=latent_frame_ids,
                    pose_mode=pose_mode,
                    inverse_action_channel_ids=inverse_action_channel_ids,
                )
                processed_segments.append(processed_actions)
                segment_count += 1
                progress_bar.update(1)

            if max_segments is not None and segment_count >= max_segments:
                break
    finally:
        progress_bar.close()

    if not processed_segments:
        raise RuntimeError("No action segments were processed.")

    all_processed_actions = np.concatenate(processed_segments, axis=0)
    summary = {
        "episode_count": len(episode_action_cache),
        "segment_count": segment_count,
        "action_step_count": int(all_processed_actions.shape[0]),
    }
    return all_processed_actions, summary


def write_statistics(
    output_path: Path,
    dataset_root: Path,
    camera_key: str,
    pose_mode: PoseMode,
    lower_quantile: float,
    upper_quantile: float,
    processed_actions: np.ndarray,
    summary: dict[str, int],
) -> None:
    q01 = np.quantile(processed_actions, lower_quantile, axis=0)
    q99 = np.quantile(processed_actions, upper_quantile, axis=0)

    statistics = {
        "dataset_root": str(dataset_root.resolve()),
        "representation": {
            "stored_action": "14d_bimanual_xyz_euler_gripper",
            "pose_mode": pose_mode,
            "model_action": "30d_canonical_bimanual_xyz_quaternion_gripper",
            "quaternion_order": "xyzw",
            "euler_axes": "xyz",
            "quaternion_conversion": "scipy.spatial.transform.Rotation.from_euler",
            "camera_key_for_frame_ids": camera_key,
            "used_action_channel_ids": DEFAULT_USED_ACTION_CHANNEL_IDS,
            "includes_training_leading_padding": True,
        },
        "quantiles": {
            "lower": lower_quantile,
            "upper": upper_quantile,
        },
        "norm_stat": {
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        },
        "summary": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(statistics, output_file, indent=2)
        output_file.write("\n")

    print(f"Wrote action normalization statistics to: {output_path}")
    print(f"Processed episodes: {summary['episode_count']}")
    print(f"Processed segments: {summary['segment_count']}")
    print(f"Processed action steps: {summary['action_step_count']}")
    print("Config-ready norm_stat:")
    print(json.dumps(statistics["norm_stat"], indent=2))


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)

    dataset_root = arguments.dataset_root.resolve()
    output_path = (
        arguments.output.resolve()
        if arguments.output is not None
        else dataset_root / "meta" / "action_norm_stats.json"
    )
    processed_actions, summary = collect_processed_actions(
        dataset_root=dataset_root,
        camera_key=arguments.camera_key,
        pose_mode=arguments.pose_mode,
        max_segments=arguments.max_segments,
    )
    write_statistics(
        output_path=output_path,
        dataset_root=dataset_root,
        camera_key=arguments.camera_key,
        pose_mode=arguments.pose_mode,
        lower_quantile=arguments.lower_quantile,
        upper_quantile=arguments.upper_quantile,
        processed_actions=processed_actions,
        summary=summary,
    )


if __name__ == "__main__":
    main()
