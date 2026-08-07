#!/usr/bin/env python3
"""Upload RoboTwin FastWAM text embeddings to ModelScope."""

from __future__ import annotations

import argparse
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from modelscope.hub.api import HubApi


DEFAULT_SOURCE_DIR = Path("/sharedata/lsy/.cache/robotwinfastwam/textembed")
DEFAULT_STAGING_DIR = Path("/sharedata/lsy/.cache/robotwinfastwam/modelscope_textembed_shards")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local text embedding cache directory to ModelScope."
    )
    parser.add_argument(
        "repo_id",
        help="Target ModelScope repo id, for example `namespace/repo_name`.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Local directory to upload. Default: {DEFAULT_SOURCE_DIR}",
    )
    parser.add_argument(
        "--path-in-repo",
        default="textembed",
        help=(
            "Target directory inside the ModelScope repo. "
            "Use an empty string to upload the source contents to repo root."
        ),
    )
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset"),
        default="dataset",
        help="ModelScope repo type. Default: dataset.",
    )
    parser.add_argument(
        "--revision",
        default="master",
        help="Target branch or revision. Default: master.",
    )
    parser.add_argument(
        "--token",
        default='ms-cd5a34e5-fe12-42d5-8edc-550f00f3b04f',
        help="ModelScope access token. Defaults to MODELSCOPE_TOKEN.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload FastWAM text embeddings",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Number of upload workers. Default: 32.",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="+",
        default=None,
        help="Only upload paths matching these patterns, for example `*.pt`.",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="+",
        default=None,
        help="Skip paths matching these patterns.",
    )
    parser.add_argument(
        "--split-top-level",
        action="store_true",
        help=(
            "Upload each top-level file/directory separately. Recommended for very "
            "large folders because failures can be resumed at a smaller unit."
        ),
    )
    parser.add_argument(
        "--parallel-batches",
        type=int,
        default=1,
        help=(
            "Number of top-level batches to upload concurrently when using "
            "--split-top-level. Start with 2 or 4 for very large uploads. Default: 1."
        ),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="With --split-top-level, upload only these top-level names.",
    )
    parser.add_argument(
        "--pack-shards",
        action="store_true",
        help=(
            "Pack many small files into tar shards and upload shard files instead. "
            "Use this when the repo file count limit is exceeded."
        ),
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=20.0,
        help="Target uncompressed payload size per tar shard. Default: 20 GB.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING_DIR,
        help=f"Directory for temporary tar shards. Default: {DEFAULT_STAGING_DIR}",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep local tar shards after successful upload.",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the target repo first if it does not exist.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="When used with --create-repo, create a private repo.",
    )
    return parser.parse_args()


def iter_top_level_entries(source_dir: Path, only: Iterable[str] | None) -> list[Path]:
    only_set = set(only or [])
    entries = sorted(
        path
        for path in source_dir.iterdir()
        if not only_set or path.name in only_set
    )
    if only_set:
        found = {path.name for path in entries}
        missing = sorted(only_set - found)
        if missing:
            raise FileNotFoundError(f"Top-level entries not found: {missing}")
    return entries


def upload_one(
    *,
    repo_id: str,
    source_path: Path,
    path_in_repo: str,
    commit_message: str,
    token: str,
    repo_type: str,
    max_workers: int,
    revision: str,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
):
    api = HubApi()
    api.login(token)
    print(f"Uploading {source_path} -> {repo_id}:{revision}/{path_in_repo}")
    return api.upload_folder(
        repo_id=repo_id,
        folder_path=source_path,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        token=token,
        repo_type=repo_type,
        max_workers=max_workers,
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def iter_files(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            yield path


def upload_file_one(
    *,
    repo_id: str,
    file_path: Path,
    path_in_repo: str,
    commit_message: str,
    token: str,
    repo_type: str,
    revision: str,
):
    api = HubApi()
    api.login(token)
    print(f"Uploading shard {file_path} -> {repo_id}:{revision}/{path_in_repo}")
    return api.upload_file(
        repo_id=repo_id,
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        token=token,
        repo_type=repo_type,
        revision=revision,
    )


def pack_and_upload_shards(
    *,
    source_dir: Path,
    staging_dir: Path,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
    token: str,
    repo_type: str,
    revision: str,
    shard_size_gb: float,
    keep_shards: bool,
):
    staging_dir.mkdir(parents=True, exist_ok=True)
    shard_size_bytes = int(shard_size_gb * 1024**3)
    if shard_size_bytes <= 0:
        raise ValueError("--shard-size-gb must be positive.")

    commit_infos = []
    shard_idx = 0
    tar = None
    tar_path = None
    shard_payload_size = 0

    def open_shard(idx: int):
        path = staging_dir / f"textembed-{idx:06d}.tar"
        print(f"Creating shard {path}")
        return path, tarfile.open(path, mode="w")

    def close_upload_cleanup():
        nonlocal tar, tar_path, shard_payload_size
        if tar is None or tar_path is None:
            return
        tar.close()
        tar = None
        shard_name = tar_path.name
        shard_repo_path = "/".join(
            part.strip("/")
            for part in (path_in_repo, "textembed-shards", shard_name)
            if part and part.strip("/")
        )
        commit_infos.append(
            upload_file_one(
                repo_id=repo_id,
                file_path=tar_path,
                path_in_repo=shard_repo_path,
                commit_message=f"{commit_message}: {shard_name}",
                token=token,
                repo_type=repo_type,
                revision=revision,
            )
        )
        if not keep_shards:
            tar_path.unlink()
        tar_path = None
        shard_payload_size = 0

    try:
        for file_path in iter_files(source_dir):
            file_size = file_path.stat().st_size
            if tar is not None and shard_payload_size > 0 and shard_payload_size + file_size > shard_size_bytes:
                close_upload_cleanup()
                shard_idx += 1

            if tar is None:
                tar_path, tar = open_shard(shard_idx)

            arcname = file_path.relative_to(source_dir).as_posix()
            tar.add(file_path, arcname=arcname, recursive=False)
            shard_payload_size += file_size

        close_upload_cleanup()
    finally:
        if tar is not None:
            tar.close()

    return commit_infos


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")
    if not args.token:
        raise ValueError(
            "Missing ModelScope token. Set MODELSCOPE_TOKEN or pass --token."
        )

    api = HubApi()
    api.login(args.token)

    if args.create_repo:
        visibility = "private" if args.private else "public"
        api.create_repo(
            args.repo_id,
            token=args.token,
            visibility=visibility,
            repo_type=args.repo_type,
            exist_ok=True,
        )

    if args.pack_shards:
        commit_infos = pack_and_upload_shards(
            source_dir=source_dir,
            staging_dir=args.staging_dir.expanduser().resolve(),
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            commit_message=args.commit_message,
            token=args.token,
            repo_type=args.repo_type,
            revision=args.revision,
            shard_size_gb=args.shard_size_gb,
            keep_shards=args.keep_shards,
        )
        print(f"Upload finished: {commit_infos}")
        return

    if args.split_top_level:
        entries = iter_top_level_entries(source_dir, args.only)

        def upload_entry(entry: Path):
            entry_path_parts = (
                (args.path_in_repo, entry.name) if entry.is_dir() else (args.path_in_repo,)
            )
            entry_path_in_repo = "/".join(
                part.strip("/") for part in entry_path_parts if part and part.strip("/")
            )
            return upload_one(
                repo_id=args.repo_id,
                source_path=entry,
                path_in_repo=entry_path_in_repo,
                commit_message=f"{args.commit_message}: {entry.name}",
                token=args.token,
                repo_type=args.repo_type,
                max_workers=args.max_workers,
                revision=args.revision,
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
            )

        if args.parallel_batches <= 1:
            commit_infos = [upload_entry(entry) for entry in entries]
        else:
            commit_infos = []
            with ThreadPoolExecutor(max_workers=args.parallel_batches) as executor:
                futures = {executor.submit(upload_entry, entry): entry for entry in entries}
                for future in as_completed(futures):
                    entry = futures[future]
                    try:
                        commit_infos.append(future.result())
                    except Exception as exc:
                        raise RuntimeError(f"Failed to upload batch `{entry.name}`") from exc

        print(f"Upload finished: {commit_infos}")
        return

    commit_info = upload_one(
        repo_id=args.repo_id,
        source_path=source_dir,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
        token=args.token,
        repo_type=args.repo_type,
        max_workers=args.max_workers,
        revision=args.revision,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )
    print(f"Upload finished: {commit_info}")


if __name__ == "__main__":
    main()
