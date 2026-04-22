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
3. iemocap: normalize IEMOCAP parquet shards plus a raw video archive into a manifest.
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
IEMOCAP_PARQUETS = (
    "train-00000-of-00003.parquet",
    "train-00001-of-00003.parquet",
    "train-00002-of-00003.parquet",
)


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

    iemocap_parser = subparsers.add_parser(
        "iemocap",
        help="Prepare IEMOCAP into standardized text/audio/video assets and a JSONL manifest.",
    )
    iemocap_parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="IEMOCAP root folder.",
    )
    iemocap_parser.add_argument(
        "--output-jsonl",
        default=None,
        type=Path,
        help="Output JSONL path. Defaults to <dataset-root>/manifests/iemocap.jsonl.",
    )
    iemocap_parser.add_argument(
        "--overwrite-audio",
        action="store_true",
        help="Rewrite wav files under media/audio/ even if they already exist.",
    )
    iemocap_parser.add_argument(
        "--overwrite-text",
        action="store_true",
        help="Rewrite text files under prepared/text/ even if they already exist.",
    )
    iemocap_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only prepare the first N samples.",
    )

    args = parser.parse_args()
    if args.mode is None:
        parser.error("Please specify a mode: generic, meld, or iemocap.")
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


def require_librosa() -> object:
    try:
        import librosa  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "This path requires librosa. Install it with `pip install librosa`."
        ) from exc
    return librosa


def require_numpy() -> object:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "This path requires numpy. Install it with `pip install numpy`."
        ) from exc
    return np


def require_pyarrow_parquet() -> object:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "This mode requires pyarrow. Install it with `pip install pyarrow`."
        ) from exc
    return pq


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
    librosa = require_librosa()
    np = require_numpy()
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(str(video_path))
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        container.close()
        raise RuntimeError(f"No audio stream found in {video_path}")

    audio_stream = audio_streams[0]
    chunks: List[np.ndarray] = []
    source_rate: Optional[int] = None

    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            array = frame.to_ndarray()
            waveform = np.asarray(array)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=0)
            else:
                waveform = waveform.reshape(-1)

            if np.issubdtype(waveform.dtype, np.integer):
                scale = max(abs(np.iinfo(waveform.dtype).min), np.iinfo(waveform.dtype).max)
                waveform = waveform.astype(np.float32) / float(scale)
            else:
                waveform = waveform.astype(np.float32)

            if not waveform.size:
                continue

            chunks.append(waveform)
            if frame.sample_rate:
                source_rate = int(frame.sample_rate)

    container.close()

    if not chunks:
        raise RuntimeError(f"No decoded audio samples found in {video_path}")

    waveform = np.concatenate(chunks)
    if source_rate is None or source_rate <= 0:
        source_rate = int(getattr(audio_stream, "rate", 16000) or 16000)
    if source_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=source_rate, target_sr=16000)

    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).astype(np.int16)

    with wave.open(str(audio_path), "wb") as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)
        wav_handle.setframerate(16000)
        wav_handle.writeframes(pcm16.tobytes())


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


def iemocap_sample_id(file_name: str) -> str:
    return f"iemocap_{Path(file_name).stem}"


def iemocap_session_name(file_name: str) -> str:
    stem = Path(file_name).stem
    return stem.split("_", 1)[0] if "_" in stem else "unknown"


def find_iemocap_parquet_files(dataset_root: Path) -> List[Path]:
    candidates: List[Path] = []
    for parquet_name in IEMOCAP_PARQUETS:
        possible_paths = [
            dataset_root / "raw" / "original" / parquet_name,
            dataset_root / "raw" / parquet_name,
            dataset_root / parquet_name,
        ]
        found = next((path.resolve() for path in possible_paths if path.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"Missing IEMOCAP parquet shard '{parquet_name}'. Expected under {dataset_root / 'raw' / 'original'}."
            )
        candidates.append(found)
    return candidates


def find_iemocap_video_archive(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "raw" / "original" / "IEMOCAP_video_selected_classes.tar.gz",
        dataset_root / "raw" / "IEMOCAP_video_selected_classes.tar.gz",
        dataset_root / "IEMOCAP_video_selected_classes.tar.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Missing IEMOCAP video archive. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def ensure_iemocap_video_root(dataset_root: Path) -> Path:
    media_root = (dataset_root / "media" / "video").resolve()
    if media_root.exists() and any(media_root.rglob("*")):
        if any(path.suffix.lower() in VIDEO_EXTENSIONS for path in media_root.rglob("*") if path.is_file()):
            return media_root

    unpacked_root = (dataset_root / "raw" / "unpacked" / "IEMOCAP_video_selected_classes").resolve()
    if not (
        unpacked_root.exists()
        and any(path.suffix.lower() in VIDEO_EXTENSIONS for path in unpacked_root.rglob("*") if path.is_file())
    ):
        archive_path = find_iemocap_video_archive(dataset_root)
        extract_tar_once(archive_path, unpacked_root)

    if unpacked_root.exists() and any(
        path.suffix.lower() in VIDEO_EXTENSIONS for path in unpacked_root.rglob("*") if path.is_file()
    ):
        return unpacked_root

    raise FileNotFoundError(f"Failed to prepare IEMOCAP videos under {dataset_root}")


def index_iemocap_videos(video_root: Path) -> Dict[str, Path]:
    videos: Dict[str, Path] = {}
    for path in sorted(video_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        videos[path.name.lower()] = path.resolve()
        videos[path.stem.lower()] = path.resolve()
    return videos


def write_audio_bytes(audio_bytes: bytes, audio_path: Path, overwrite: bool = False) -> None:
    if audio_path.exists() and not overwrite:
        return
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(audio_bytes)


def build_iemocap_records(
    dataset_root: Path,
    output_jsonl: Path,
    limit: Optional[int],
    overwrite_audio: bool,
    overwrite_text: bool,
) -> Tuple[List[dict], List[str]]:
    pq = require_pyarrow_parquet()
    parquet_files = find_iemocap_parquet_files(dataset_root)
    video_root = ensure_iemocap_video_root(dataset_root)
    video_index = index_iemocap_videos(video_root)

    prepared_text_root = (dataset_root / "prepared" / "text").resolve()
    prepared_audio_root = (dataset_root / "media" / "audio").resolve()
    prepared_text_root.mkdir(parents=True, exist_ok=True)
    prepared_audio_root.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    warnings: List[str] = []
    emotion_score_keys = [
        "frustrated",
        "angry",
        "sad",
        "disgust",
        "excited",
        "fear",
        "neutral",
        "surprise",
        "happy",
    ]

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                file_name = str(row.get("file") or "").strip()
                if not file_name:
                    warnings.append(f"Skipping row without audio file name in {parquet_path.name}.")
                    continue

                session_name = iemocap_session_name(file_name)
                sample_id = iemocap_sample_id(file_name)
                sample_text = str(row.get("transcription") or "").strip()
                label = str(row.get("major_emotion") or "").strip()
                video_name = str(row.get("video") or "").strip()
                audio_entry = row.get("audio") or {}
                audio_bytes = audio_entry.get("bytes")

                if not sample_text:
                    warnings.append(f"Skipping {sample_id}: empty transcription.")
                    continue
                if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
                    warnings.append(f"Skipping {sample_id}: missing embedded wav bytes.")
                    continue
                if not video_name:
                    warnings.append(f"Skipping {sample_id}: missing video filename.")
                    continue

                video_lookup_key = video_name.lower()
                video_path = video_index.get(video_lookup_key) or video_index.get(Path(video_name).stem.lower())
                if video_path is None:
                    warnings.append(f"Missing video for {sample_id}: expected {video_name} under {video_root}")
                    continue

                session_text_dir = prepared_text_root / session_name
                session_audio_dir = prepared_audio_root / session_name
                session_text_dir.mkdir(parents=True, exist_ok=True)
                session_audio_dir.mkdir(parents=True, exist_ok=True)

                text_path = (session_text_dir / f"{sample_id}.txt").resolve()
                if overwrite_text or not text_path.exists():
                    text_path.write_text(sample_text + "\n", encoding="utf-8")

                audio_path = (session_audio_dir / f"{sample_id}.wav").resolve()
                write_audio_bytes(bytes(audio_bytes), audio_path, overwrite=overwrite_audio)

                emotion_scores = {
                    key: float(row[key]) if row.get(key) is not None else None
                    for key in emotion_score_keys
                }

                records.append(
                    {
                        "id": sample_id,
                        "text": sample_text,
                        "audio_path": str(audio_path),
                        "video_path": str(video_path),
                        "label": label,
                        "meta": {
                            "dataset": "IEMOCAP",
                            "session": session_name,
                            "gender": row.get("gender", ""),
                            "source_audio_file": file_name,
                            "source_video_file": video_name,
                            "text_path": str(text_path),
                            "emotion_scores": emotion_scores,
                            "emo_act": row.get("EmoAct"),
                            "emo_val": row.get("EmoVal"),
                            "emo_dom": row.get("EmoDom"),
                            "speaking_rate": row.get("speaking_rate"),
                            "pitch_mean": row.get("pitch_mean"),
                            "pitch_std": row.get("pitch_std"),
                            "rms": row.get("rms"),
                            "relative_db": row.get("relative_db"),
                        },
                    }
                )

                if limit is not None and len(records) >= limit:
                    return records, warnings

    return records, warnings


def resolve_iemocap_output_path(args: argparse.Namespace, dataset_root: Path) -> Path:
    if args.output_jsonl is not None:
        return args.output_jsonl.resolve()
    return (dataset_root / "manifests" / "iemocap.jsonl").resolve()


def run_iemocap_mode(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    output_jsonl = resolve_iemocap_output_path(args, dataset_root)

    records, warnings = build_iemocap_records(
        dataset_root=dataset_root,
        output_jsonl=output_jsonl,
        limit=args.limit,
        overwrite_audio=args.overwrite_audio,
        overwrite_text=args.overwrite_text,
    )
    write_jsonl(records, output_jsonl)
    warning_path = warning_path_for_output(output_jsonl)
    write_warnings(warnings, warning_path)

    print(f"IEMOCAP manifest written to: {output_jsonl}")
    print(f"Prepared IEMOCAP samples: {len(records)}")
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
    if args.mode == "iemocap":
        run_iemocap_mode(args)
        return
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
