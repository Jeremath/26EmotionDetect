#!/usr/bin/env python3
"""
Build a JSONL manifest from a dataset folder with:
DESFOLDER/
  text/
  audio/
  video/

Usage:
    python data_process.py --DESFOLDER /path/to/dataset --OUTFOLDER /path/to/output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TEXT_EXTENSIONS = {".txt", ".text"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DESFOLDER/text, audio, video into a JSONL manifest."
    )
    parser.add_argument(
        "--DESFOLDER",
        required=True,
        type=Path,
        help="Dataset root folder containing text/, audio/, and video/.",
    )
    parser.add_argument(
        "--OUTFOLDER",
        required=True,
        type=Path,
        help="Output folder where samples.jsonl will be written.",
    )
    return parser.parse_args()


def collect_files(folder: Path, extensions: Sequence[str]) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        files[path.stem] = path.resolve()
    return files


def ensure_required_dirs(dataset_root: Path) -> Tuple[Path, Path, Path]:
    text_dir = dataset_root / "text"
    audio_dir = dataset_root / "audio"
    video_dir = dataset_root / "video"

    missing = [str(path) for path in (text_dir, audio_dir, video_dir) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required subfolders under DESFOLDER: " + ", ".join(missing)
        )

    return text_dir, audio_dir, video_dir


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_manifest_records(
    text_files: Dict[str, Path],
    audio_files: Dict[str, Path],
    video_files: Dict[str, Path],
) -> Tuple[List[dict], List[str]]:
    records: List[dict] = []
    warnings: List[str] = []

    shared_ids = sorted(set(text_files) & set(audio_files) & set(video_files))

    if not shared_ids:
        warnings.append("No matched samples found across text/, audio/, and video/.")

    missing_text = sorted((set(audio_files) | set(video_files)) - set(text_files))
    missing_audio = sorted((set(text_files) | set(video_files)) - set(audio_files))
    missing_video = sorted((set(text_files) | set(audio_files)) - set(video_files))

    for sample_id in missing_text:
        warnings.append(f"Skipped '{sample_id}': missing text file.")
    for sample_id in missing_audio:
        warnings.append(f"Skipped '{sample_id}': missing audio file.")
    for sample_id in missing_video:
        warnings.append(f"Skipped '{sample_id}': missing video file.")

    for sample_id in shared_ids:
        text_path = text_files[sample_id]
        audio_path = audio_files[sample_id]
        video_path = video_files[sample_id]

        records.append(
            {
                "id": sample_id,
                "text": read_text_file(text_path),
                "audio_path": str(audio_path),
                "video_path": str(video_path),
            }
        )

    return records, warnings


def write_jsonl(records: Iterable[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    dataset_root = args.DESFOLDER.resolve()
    output_root = args.OUTFOLDER.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    text_dir, audio_dir, video_dir = ensure_required_dirs(dataset_root)
    text_files = collect_files(text_dir, TEXT_EXTENSIONS)
    audio_files = collect_files(audio_dir, AUDIO_EXTENSIONS)
    video_files = collect_files(video_dir, VIDEO_EXTENSIONS)

    records, warnings = build_manifest_records(text_files, audio_files, video_files)

    manifest_path = output_root / "samples.jsonl"
    write_jsonl(records, manifest_path)

    print(f"Manifest written to: {manifest_path}")
    print(f"Matched samples: {len(records)}")

    if warnings:
        warning_path = output_root / "data_process_warnings.log"
        warning_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        print(f"Warnings written to: {warning_path}")


if __name__ == "__main__":
    main()
