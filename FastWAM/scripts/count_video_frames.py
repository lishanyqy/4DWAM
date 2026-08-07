#!/usr/bin/env python3
"""Count frames for all videos under a directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import av


DEFAULT_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count video frames under a directory and print a JSON list."
    )
    parser.add_argument("directory", type=Path, help="Directory containing videos.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(DEFAULT_EXTENSIONS),
        help="Video extensions to include. Default: common video formats.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan files directly under directory.",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Decode every video to count frames exactly instead of trusting metadata.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Print paths relative to the input directory.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for one-line output.",
    )
    return parser.parse_args()


def iter_video_paths(directory: Path, extensions: Iterable[str], recursive: bool) -> list[Path]:
    normalized_exts = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
    }
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in normalized_exts
    )


def count_frames(path: Path, decode: bool) -> int:
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError("no video stream found")

        if not decode and stream.frames:
            return int(stream.frames)

        return sum(1 for _ in container.decode(stream))


def main() -> None:
    args = parse_args()
    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Directory does not exist: {directory}")

    results = []
    for path in iter_video_paths(
        directory,
        extensions=args.extensions,
        recursive=not args.no_recursive,
    ):
        output_path = str(path.relative_to(directory) if args.relative else path)
        item = {"path": output_path}
        try:
            item["frames"] = count_frames(path, decode=args.decode)
        except Exception as exc:  # Keep scanning even if one file is broken.
            item["frames"] = None
            item["error"] = str(exc)
        results.append(item)

    indent = None if args.indent == 0 else args.indent
    print(json.dumps(results, ensure_ascii=False, indent=indent))


if __name__ == "__main__":
    main()
