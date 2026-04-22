#!/usr/bin/env python3
"""
Dataset downloader for multimodal emotion recognition experiments.

Design goals:
1. Keep a standard on-disk layout under data/<dataset_name>/.
2. Make new datasets easy to add through a small registry.
3. Separate downloading from preprocessing, while still supporting archive extraction.

Standard layout:
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
      dataset_info.json
      README_DATASET.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


CHUNK_SIZE = 1024 * 1024
MELD_SPLITS = ("train", "dev", "test")
IEMOCAP_PARQUETS = (
    "train-00000-of-00003.parquet",
    "train-00001-of-00003.parquet",
    "train-00002-of-00003.parquet",
)


@dataclass(frozen=True)
class DownloadItem:
    relative_path: str
    urls: List[str]
    description: str
    extract: bool = False
    nested_split_extract: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    storage_dir: str
    display_name: str
    description: str
    homepage: str
    downloads: List[DownloadItem] = field(default_factory=list)


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "meld": DatasetSpec(
        key="meld",
        storage_dir="MELD",
        display_name="MELD",
        description="Multimodal EmotionLines Dataset for conversational emotion recognition.",
        homepage="https://github.com/declare-lab/MELD",
        downloads=[
            DownloadItem(
                relative_path="annotations/train_sent_emo.csv",
                urls=[
                    "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
                ],
                description="MELD train annotations",
            ),
            DownloadItem(
                relative_path="annotations/dev_sent_emo.csv",
                urls=[
                    "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
                ],
                description="MELD dev annotations",
            ),
            DownloadItem(
                relative_path="annotations/test_sent_emo.csv",
                urls=[
                    "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
                ],
                description="MELD test annotations",
            ),
            DownloadItem(
                relative_path="raw/original/MELD.Raw.tar.gz",
                urls=[
                    "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz",
                    "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz",
                ],
                description="MELD raw video archive",
                extract=True,
                nested_split_extract=True,
            ),
        ],
    ),
    "iemocap": DatasetSpec(
        key="iemocap",
        storage_dir="IEMOCAP",
        display_name="IEMOCAP",
        description="Interactive Emotional Dyadic Motion Capture dataset mirrored as parquet audio/text labels plus a raw video archive.",
        homepage="https://sail.usc.edu/iemocap/",
        downloads=[
            *[
                DownloadItem(
                    relative_path=f"raw/original/{parquet_name}",
                    urls=[
                        f"https://huggingface.co/datasets/WiktorJakubowski/iemocap-with-videos/resolve/main/data/{parquet_name}",
                    ],
                    description=f"IEMOCAP parquet shard {parquet_name}",
                )
                for parquet_name in IEMOCAP_PARQUETS
            ],
            DownloadItem(
                relative_path="raw/original/IEMOCAP_video_selected_classes.tar.gz",
                urls=[
                    "https://huggingface.co/datasets/tarasabkar/IEMOCAP_videos/resolve/main/IEMOCAP_videos",
                ],
                description="IEMOCAP raw video archive",
                extract=True,
            ),
        ],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and optionally extract multimodal datasets."
    )
    parser.add_argument(
        "--dataset",
        default="meld",
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Dataset key defined in DATASET_REGISTRY.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory used to store datasets.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist on disk.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download archives/files without extracting them.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Redownload files even if they already exist.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List supported datasets and exit.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha1_of_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def download_file(urls: List[str], target_path: Path, skip_existing: bool, force_redownload: bool) -> Path:
    ensure_parent(target_path)

    if target_path.exists() and skip_existing and not force_redownload:
        print(f"Skip existing: {target_path}")
        return target_path

    last_error: Optional[Exception] = None
    for url in urls:
        tmp_path = target_path.with_suffix(target_path.suffix + ".part")
        if tmp_path.exists():
            tmp_path.unlink()

        print(f"Downloading: {url}")
        print(f"      into: {target_path}")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request) as response, tmp_path.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
            tmp_path.replace(target_path)
            file_size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"Downloaded {file_size_mb:.2f} MB, sha1={sha1_of_file(target_path)}")
            return target_path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"Download failed from {url}: {exc}")

    raise RuntimeError(f"All download sources failed for {target_path}") from last_error


def find_nested_archives(root_dir: Path) -> Iterable[Path]:
    for pattern in ("*.tar.gz", "*.tgz", "*.tar"):
        yield from root_dir.rglob(pattern)


def find_meld_split_archive(unpacked_root: Path, split: str) -> Path:
    archive = next((path for path in unpacked_root.rglob(f"{split}.tar.gz")), None)
    if archive is None:
        raise FileNotFoundError(f"Could not find MELD nested archive for split '{split}' under {unpacked_root}")
    return archive


def extract_meld_split_archives(dataset_dir: Path, unpacked_root: Path) -> None:
    for split in MELD_SPLITS:
        split_archive = find_meld_split_archive(unpacked_root, split)
        split_dir = dataset_dir / "media" / "video" / split
        extract_tar_once(split_archive, split_dir)


def extract_downloaded_item(dataset_dir: Path, archive_path: Path, item: DownloadItem) -> None:
    unpacked_root = dataset_dir / "raw" / "unpacked" / archive_path.stem.replace(".tar", "")
    extract_tar_once(archive_path, unpacked_root)

    if item.nested_split_extract and dataset_dir.name == DATASET_REGISTRY["meld"].storage_dir:
        extract_meld_split_archives(dataset_dir, unpacked_root)


def ensure_standard_dirs(dataset_dir: Path) -> None:
    for relative in (
        "annotations",
        "raw/original",
        "raw/unpacked",
        "media/video",
        "media/audio",
        "prepared/text",
        "manifests",
    ):
        (dataset_dir / relative).mkdir(parents=True, exist_ok=True)


def write_dataset_metadata(dataset_dir: Path, spec: DatasetSpec) -> None:
    metadata = {
        "dataset_key": spec.key,
        "display_name": spec.display_name,
        "description": spec.description,
        "homepage": spec.homepage,
        "storage_dir": spec.storage_dir,
        "layout": {
            "annotations": "annotations/",
            "raw_original": "raw/original/",
            "raw_unpacked": "raw/unpacked/",
            "media_video": "media/video/",
            "media_audio": "media/audio/",
            "prepared_text": "prepared/text/",
            "manifests": "manifests/",
        },
    }
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    notes = [
        f"Dataset: {spec.display_name}",
        f"Registry key: {spec.key}",
        f"Homepage: {spec.homepage}",
        "",
        "Standard layout:",
        "- annotations/: labels, transcripts, or split metadata",
        "- raw/original/: downloaded archives",
        "- raw/unpacked/: extracted raw archives",
        "- media/video/: standardized extracted videos, typically split by train/dev/test",
        "- media/audio/: standardized extracted audio files",
        "- prepared/text/: normalized text files",
        "- manifests/: JSONL manifests for downstream pipelines",
        "",
        "This directory is managed by src/data_require.py and src/data_process.py.",
        "",
        "To add another dataset later:",
        "1. Add a DatasetSpec entry to DATASET_REGISTRY in src/data_require.py.",
        "2. Add a corresponding preprocessing mode in src/data_process.py.",
        "3. Keep the same standard layout so the main inference pipeline can reuse the output manifest.",
    ]
    (dataset_dir / "README_DATASET.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


def list_datasets() -> None:
    for key, spec in sorted(DATASET_REGISTRY.items()):
        print(f"{key}: {spec.display_name}")
        print(f"  storage: data/{spec.storage_dir}")
        print(f"  description: {spec.description}")
        print(f"  homepage: {spec.homepage}")


def main() -> None:
    args = parse_args()
    if args.list_datasets:
        list_datasets()
        return

    spec = DATASET_REGISTRY[args.dataset]
    dataset_dir = (args.data_root.resolve() / spec.storage_dir).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ensure_standard_dirs(dataset_dir)

    print(f"Dataset key: {spec.key}")
    print(f"Dataset name: {spec.display_name}")
    print(f"Description: {spec.description}")
    print(f"Target directory: {dataset_dir}")

    for item in spec.downloads:
        target_path = dataset_dir / item.relative_path
        archive_path = download_file(
            urls=item.urls,
            target_path=target_path,
            skip_existing=args.skip_existing,
            force_redownload=args.force_redownload,
        )
        if item.extract and not args.download_only:
            extract_downloaded_item(dataset_dir=dataset_dir, archive_path=archive_path, item=item)

    write_dataset_metadata(dataset_dir, spec)
    print("Finished.")


if __name__ == "__main__":
    main()
