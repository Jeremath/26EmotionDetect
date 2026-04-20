#!/usr/bin/env python3
"""
Build standardized JSONL manifests for multimodal emotion datasets.

The output manifest is designed to be consumed directly by:
    - src/multimodal_emotion_pipeline.py
    - src/debate.py

Standard layout per dataset:
    <dataset_root>/
      annotations/
      raw/
        original/
        unpacked/
      media/
        video/
        audio/
      prepared/
        text/
      manifests/

Supported modes:
1. generic: convert an existing text/audio/video directory into a manifest.
2. meld: normalize MELD into separate text/audio/video assets plus a manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tarfile
import wave
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


TEXT_EXTENSIONS = {".txt", ".text"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
MELD_SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert multimodal datasets into standardized JSONL manifests."
    )
    subparsers = parser.add_subparsers(dest="mode")

    generic_parser = subparsers.add_parser(
        "generic",
        help="Convert dataset text/audio/video folders into a standardized manifest.",
    )
    generic_parser.add_argument(
        "--dataset-root",
        "--DESFOLDER",
        dest="dataset_root",
        required=True,
        type=Path,
        help="Dataset root folder containing text/, audio/, and video/.",
    )
    generic_parser.add_argument(
        "--output-root",
        "--OUTFOLDER",
        dest="output_root",
        default=None,
        type=Path,
        help="Optional output root. If set and --output-jsonl is omitted, writes output_root/samples.jsonl.",
    )
    generic_parser.add_argument(
        "--output-jsonl",
        default=None,
        type=Path,
        help="Explicit output JSONL path. Defaults to <dataset-root>/manifests/samples.jsonl.",
    )
    generic_parser.add_argument(
        "--text-dir",
        default="text",
        help="Text subdirectory under dataset-root.",
    )
    generic_parser.add_argument(
        "--audio-dir",
        default="audio",
        help="Audio subdirectory under dataset-root.",
    )
    generic_parser.add_argument(
        "--video-dir",
        default="video",
        help="Video subdirectory under dataset-root.",
    )

    meld_parser = subparsers.add_parser(
        "meld",
        help="Prepare MELD into standardized text/audio/video assets and a JSONL manifest.",
    )
    meld_parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="MELD root folder.",
    )
    meld_parser.add_argument(
        "--output-jsonl",
        default=None,
        type=Path,
        help="Output JSONL path. Defaults to <dataset-root>/manifests/meld.jsonl.",
    )
    meld_parser.add_argument(
        "--audio-source",
        choices=["extract", "video"],
        default="extract",
        help="How to populate audio_path in the manifest. 'extract' creates wav files under media/audio/. 'video' points audio_path to the mp4 clip.",
    )
    meld_parser.add_argument(
        "--extract-audio",
        action="store_true",
        help="Legacy compatibility flag. Equivalent to --audio-source extract.",
    )
    meld_parser.add_argument(
        "--overwrite-audio",
        action="store_true",
        help="Re-extract wav audio files even if they already exist.",
    )
    meld_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only prepare the first N samples across all splits.",
    )

    args = parser.parse_args()
    if args.mode is None:
        parser.error("Please specify a mode: generic or meld.")
    if getattr(args, "extract_audio", False):
        args.audio_source = "extract"
    return args


def warning_path_for_output(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_data_process_warnings.log"


def write_jsonl(records: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_warnings(warnings: List[str], warning_path: Path) -> None:
    if not warnings:
        if warning_path.exists():
            warning_path.unlink()
        return
    warning_path.parent.mkdir(parents=True, exist_ok=True)
    warning_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def normalize_relative_key(path: Path, base_dir: Path) -> str:
    relative = path.resolve().relative_to(base_dir.resolve())
    without_suffix = relative.with_suffix("")
    return without_suffix.as_posix()


def collect_files(folder: Path, extensions: Sequence[str]) -> Tuple[Dict[str, Path], List[str]]:
    files: Dict[str, Path] = {}
    warnings: List[str] = []

    if not folder.exists():
        raise FileNotFoundError(f"Required folder does not exist: {folder}")

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue

        key = normalize_relative_key(path, folder)
        if key in files and files[key] != path.resolve():
            warnings.append(
                f"Duplicate key '{key}' under {folder}; keeping {files[key]} and skipping {path.resolve()}."
            )
            continue
        files[key] = path.resolve()

    return files, warnings


def build_manifest_records(
    text_files: Dict[str, Path],
    audio_files: Dict[str, Path],
    video_files: Dict[str, Path],
) -> Tuple[List[dict], List[str]]:
    records: List[dict] = []
    warnings: List[str] = []

    shared_ids = sorted(set(text_files) & set(audio_files) & set(video_files))
    missing_text = sorted((set(audio_files) | set(video_files)) - set(text_files))
    missing_audio = sorted((set(text_files) | set(video_files)) - set(audio_files))
    missing_video = sorted((set(text_files) | set(audio_files)) - set(video_files))

    if not shared_ids:
        warnings.append("No matched samples found across text, audio, and video folders.")

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
                "id": sample_id.replace("/", "_"),
                "text": read_text_file(text_path),
                "audio_path": str(audio_path),
                "video_path": str(video_path),
                "meta": {
                    "dataset": "generic",
                    "relative_key": sample_id,
                    "text_path": str(text_path),
                },
            }
        )

    return records, warnings


def resolve_generic_output_path(args: argparse.Namespace, dataset_root: Path) -> Path:
    if args.output_jsonl is not None:
        return args.output_jsonl.resolve()
    if args.output_root is not None:
        return (args.output_root.resolve() / "samples.jsonl").resolve()
    return (dataset_root / "manifests" / "samples.jsonl").resolve()


def run_generic_mode(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    text_dir = (dataset_root / args.text_dir).resolve()
    audio_dir = (dataset_root / args.audio_dir).resolve()
    video_dir = (dataset_root / args.video_dir).resolve()
    output_jsonl = resolve_generic_output_path(args, dataset_root)

    text_files, text_warnings = collect_files(text_dir, TEXT_EXTENSIONS)
    audio_files, audio_warnings = collect_files(audio_dir, AUDIO_EXTENSIONS)
    video_files, video_warnings = collect_files(video_dir, VIDEO_EXTENSIONS)

    records, warnings = build_manifest_records(text_files, audio_files, video_files)
    warnings = text_warnings + audio_warnings + video_warnings + warnings

    write_jsonl(records, output_jsonl)
    warning_path = warning_path_for_output(output_jsonl)
    write_warnings(warnings, warning_path)

    print(f"Manifest written to: {output_jsonl}")
    print(f"Matched samples: {len(records)}")
    if warnings:
        print(f"Warnings written to: {warning_path}")


def require_pyav() -> object:
    try:
        import av  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "This mode requires PyAV. Install it with `pip install av`."
        ) from exc
    return av


def safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = (target_dir / member.name).resolve()
            if os.path.commonpath([str(target_root), str(member_path)]) != str(target_root):
                raise ValueError(f"Unsafe path detected in archive: {member.name}")
        archive.extractall(target_dir)


def extract_tar_once(archive_path: Path, target_dir: Path) -> None:
    marker = target_dir / ".extract_complete"
    if marker.exists():
        return
    safe_extract_tar(archive_path, target_dir)
    marker.write_text("ok\n", encoding="utf-8")


def find_meld_raw_archive(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "raw" / "original" / "MELD.Raw.tar.gz",
        dataset_root / "raw" / "MELD.Raw.tar.gz",
        dataset_root / "downloads" / "MELD.Raw.tar.gz",
        dataset_root / "MELD.Raw.tar.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Missing MELD raw archive. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def find_meld_annotation_dir(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "annotations",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing MELD annotations directory under {dataset_root}")


def find_meld_split_archive(unpacked_root: Path, split: str) -> Path:
    archive = next((path for path in unpacked_root.rglob(f"{split}.tar.gz")), None)
    if archive is None:
        raise FileNotFoundError(f"Could not find MELD nested archive for split '{split}' under {unpacked_root}")
    return archive


def ensure_meld_video_dirs(dataset_root: Path) -> Dict[str, Path]:
    standard_dirs = {split: (dataset_root / "media" / "video" / split).resolve() for split in MELD_SPLITS}
    if all(path.exists() and any(path.rglob("*.mp4")) for path in standard_dirs.values()):
        return standard_dirs

    legacy_dirs = {split: (dataset_root / "raw" / "unpacked" / split).resolve() for split in MELD_SPLITS}
    try:
        archive_path = find_meld_raw_archive(dataset_root)
    except FileNotFoundError:
        if all(path.exists() and any(path.rglob("*.mp4")) for path in legacy_dirs.values()):
            return legacy_dirs
        raise

    unpacked_root = (dataset_root / "raw" / "unpacked" / "MELD.Raw").resolve()
    extract_tar_once(archive_path, unpacked_root)

    for split in MELD_SPLITS:
        split_archive = find_meld_split_archive(unpacked_root, split)
        split_dir = standard_dirs[split]
        extract_tar_once(split_archive, split_dir)

    if all(path.exists() and any(path.rglob("*.mp4")) for path in standard_dirs.values()):
        return standard_dirs
    if all(path.exists() and any(path.rglob("*.mp4")) for path in legacy_dirs.values()):
        return legacy_dirs
    raise FileNotFoundError("Failed to prepare MELD split video directories.")


def index_meld_videos(split_dir: Path) -> Dict[str, Path]:
    videos: Dict[str, Path] = {}
    for path in split_dir.rglob("*.mp4"):
        videos[path.name.lower()] = path.resolve()
    return videos


def iter_csv_rows(csv_path: Path) -> Iterator[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {key.strip(): (value.strip() if isinstance(value, str) else value) for key, value in row.items()}


def meld_sample_id(split: str, dialogue_id: str, utterance_id: str) -> str:
    return f"meld_{split}_dia{dialogue_id}_utt{utterance_id}"


def meld_video_filename(dialogue_id: str, utterance_id: str) -> str:
    return f"dia{dialogue_id}_utt{utterance_id}.mp4"


def extract_audio_to_wav(video_path: Path, audio_path: Path, overwrite: bool = False) -> None:
    if audio_path.exists() and not overwrite:
        return

    av = require_pyav()
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(str(video_path))
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        container.close()
        raise RuntimeError(f"No audio stream found in {video_path}")

    audio_stream = audio_streams[0]
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=16000,
    )

    with wave.open(str(audio_path), "wb") as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)
        wav_handle.setframerate(16000)

        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                resampled = resampler.resample(frame)
                if resampled is None:
                    continue
                frames = resampled if isinstance(resampled, list) else [resampled]
                for out_frame in frames:
                    array = out_frame.to_ndarray()
                    wav_handle.writeframes(array.T.astype("int16").tobytes())
    container.close()


def build_meld_records(
    dataset_root: Path,
    output_jsonl: Path,
    limit: Optional[int],
    audio_source: str,
    overwrite_audio: bool,
) -> Tuple[List[dict], List[str]]:
    annotation_dir = find_meld_annotation_dir(dataset_root)
    split_video_dirs = ensure_meld_video_dirs(dataset_root)
    video_indices = {split: index_meld_videos(path) for split, path in split_video_dirs.items()}

    prepared_text_root = (dataset_root / "prepared" / "text").resolve()
    prepared_audio_root = (dataset_root / "media" / "audio").resolve()
    manifests_root = output_jsonl.parent.resolve()
    prepared_text_root.mkdir(parents=True, exist_ok=True)
    prepared_audio_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    warnings: List[str] = []

    for split in MELD_SPLITS:
        csv_path = annotation_dir / f"{split}_sent_emo.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing MELD annotation file: {csv_path}")

        split_text_dir = prepared_text_root / split
        split_audio_dir = prepared_audio_root / split
        split_text_dir.mkdir(parents=True, exist_ok=True)
        split_audio_dir.mkdir(parents=True, exist_ok=True)

        for row in iter_csv_rows(csv_path):
            dialogue_id = row["Dialogue_ID"]
            utterance_id = row["Utterance_ID"]
            sample_id = meld_sample_id(split, dialogue_id, utterance_id)
            video_name = meld_video_filename(dialogue_id, utterance_id).lower()
            video_path = video_indices[split].get(video_name)

            if video_path is None:
                warnings.append(f"Missing video for {sample_id}: expected {video_name} under split {split}")
                continue

            sample_text = row["Utterance"]
            text_path = (split_text_dir / f"{sample_id}.txt").resolve()
            if not text_path.exists():
                text_path.write_text(sample_text + "\n", encoding="utf-8")

            if audio_source == "extract":
                audio_path = (split_audio_dir / f"{sample_id}.wav").resolve()
                try:
                    extract_audio_to_wav(video_path, audio_path, overwrite=overwrite_audio)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"Audio extraction failed for {sample_id}: {exc}")
                    continue
                resolved_audio_path = str(audio_path)
            else:
                resolved_audio_path = str(video_path)

            records.append(
                {
                    "id": sample_id,
                    "text": sample_text,
                    "audio_path": resolved_audio_path,
                    "video_path": str(video_path),
                    "label": row.get("Emotion", ""),
                    "meta": {
                        "dataset": "MELD",
                        "split": split,
                        "speaker": row.get("Speaker", ""),
                        "sentiment": row.get("Sentiment", ""),
                        "dialogue_id": dialogue_id,
                        "utterance_id": utterance_id,
                        "season": row.get("Season", ""),
                        "episode": row.get("Episode", ""),
                        "start_time": row.get("StartTime", ""),
                        "end_time": row.get("EndTime", ""),
                        "sr_no": row.get("Sr No.", ""),
                        "text_path": str(text_path),
                        "audio_source": audio_source,
                    },
                }
            )

            if limit is not None and len(records) >= limit:
                return records, warnings

    return records, warnings


def resolve_meld_output_path(args: argparse.Namespace, dataset_root: Path) -> Path:
    if args.output_jsonl is not None:
        return args.output_jsonl.resolve()
    return (dataset_root / "manifests" / "meld.jsonl").resolve()


def run_meld_mode(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    output_jsonl = resolve_meld_output_path(args, dataset_root)

    records, warnings = build_meld_records(
        dataset_root=dataset_root,
        output_jsonl=output_jsonl,
        limit=args.limit,
        audio_source=args.audio_source,
        overwrite_audio=args.overwrite_audio,
    )
    write_jsonl(records, output_jsonl)
    warning_path = warning_path_for_output(output_jsonl)
    write_warnings(warnings, warning_path)

    print(f"MELD manifest written to: {output_jsonl}")
    print(f"Prepared MELD samples: {len(records)}")
    print(f"Audio source mode: {args.audio_source}")
    if warnings:
        print(f"Warnings written to: {warning_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "generic":
        run_generic_mode(args)
        return
    if args.mode == "meld":
        run_meld_mode(args)
        return
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
