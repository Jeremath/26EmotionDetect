#!/usr/bin/env python3
"""
Download example datasets into the local data/ directory.

Default example:
    python data_require.py

Explicit usage:
    python data_require.py --dataset meld --data-root data

What it does for MELD:
1. Download MELD annotation CSV files.
2. Download the official MELD raw archive.
3. Extract the raw archive.
4. Optionally extract nested train/dev/test tar.gz archives into videos/.

To switch to another dataset later:
1. Add a new entry into DATASET_REGISTRY.
2. Fill in its download URLs and extraction rule.
3. Run: python data_require.py --dataset <new_name>
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class DownloadItem:
    url: str
    relative_path: str
    extract: bool = False
    recursive_extract: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    downloads: List[DownloadItem] = field(default_factory=list)


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "meld": DatasetSpec(
        name="MELD",
        description="Multimodal EmotionLines Dataset for conversational emotion recognition.",
        downloads=[
            DownloadItem(
                url="https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
                relative_path="annotations/train_sent_emo.csv",
            ),
            DownloadItem(
                url="https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
                relative_path="annotations/dev_sent_emo.csv",
            ),
            DownloadItem(
                url="https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
                relative_path="annotations/test_sent_emo.csv",
            ),
            DownloadItem(
                url="http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz",
                relative_path="downloads/MELD.Raw.tar.gz",
                extract=True,
                recursive_extract=True,
            ),
        ],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets into the local data directory.")
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
        help="Skip downloading files that already exist.",
    )
    parser.add_argument(
        "--no-recursive-extract",
        action="store_true",
        help="Disable nested extraction for archives like train.tar.gz/dev.tar.gz/test.tar.gz.",
    )
    return parser.parse_args()


def print_header(dataset_key: str, spec: DatasetSpec, dataset_dir: Path) -> None:
    print(f"Dataset key: {dataset_key}")
    print(f"Dataset name: {spec.name}")
    print(f"Description: {spec.description}")
    print(f"Target directory: {dataset_dir}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha1_of_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target_path: Path, skip_existing: bool) -> Path:
    ensure_parent(target_path)

    if skip_existing and target_path.exists():
        print(f"Skip existing: {target_path}")
        return target_path

    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    print(f"Downloading: {url}")
    print(f"      into: {target_path}")

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
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


def safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = (target_dir / member.name).resolve()
            if os.path.commonpath([str(target_dir.resolve()), str(member_path)]) != str(target_dir.resolve()):
                raise ValueError(f"Unsafe path detected in archive: {member.name}")
        archive.extractall(target_dir)


def strip_archive_suffixes(path: Path) -> str:
    name = path.name
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip", ".gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def find_nested_archives(root_dir: Path) -> Iterable[Path]:
    for pattern in ("*.tar.gz", "*.tgz", "*.tar"):
        yield from root_dir.rglob(pattern)


def extract_archive(archive_path: Path, dataset_dir: Path, recursive_extract: bool) -> None:
    extracted_root = dataset_dir / "raw"
    print(f"Extracting archive: {archive_path}")
    safe_extract_tar(archive_path, extracted_root)

    if not recursive_extract:
        return

    nested_archives = [
        path
        for path in find_nested_archives(extracted_root)
        if path.resolve() != archive_path.resolve()
    ]

    if not nested_archives:
        return

    videos_root = dataset_dir / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)

    for nested_archive in nested_archives:
        split_name = strip_archive_suffixes(nested_archive)
        split_dir = videos_root / split_name
        if split_dir.exists() and any(split_dir.iterdir()):
            print(f"Skip nested extract, already exists: {split_dir}")
            continue
        print(f"Extracting nested split archive: {nested_archive} -> {split_dir}")
        safe_extract_tar(nested_archive, split_dir)


def write_notes(dataset_dir: Path, dataset_key: str, spec: DatasetSpec) -> None:
    notes_path = dataset_dir / "README_DATASET.txt"
    notes = [
        f"Dataset: {spec.name}",
        f"Registry key: {dataset_key}",
        "",
        "Downloaded by data_require.py.",
        "",
        "Directory overview:",
        "- annotations/: CSV annotation files",
        "- downloads/: original downloaded archives",
        "- raw/: extracted top-level raw archive contents",
        "- videos/: nested split archives extracted here when available",
        "",
        "MELD note:",
        "- MELD raw data are video clips in mp4 format.",
        "- If your pipeline needs separate audio files, you can later extract audio from the mp4 clips.",
        "",
        "How to add another dataset:",
        "1. Open data_require.py.",
        "2. Add a new DatasetSpec entry into DATASET_REGISTRY.",
        "3. Provide its annotation/video archive URLs.",
        "4. Run: python data_require.py --dataset <new_key>",
    ]
    notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_key = args.dataset.lower()
    spec = DATASET_REGISTRY[dataset_key]
    dataset_dir = args.data_root.resolve() / spec.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print_header(dataset_key, spec, dataset_dir)

    for item in spec.downloads:
        target_path = dataset_dir / item.relative_path
        archive_path = download_file(item.url, target_path, args.skip_existing)
        if item.extract:
            extract_archive(
                archive_path=archive_path,
                dataset_dir=dataset_dir,
                recursive_extract=item.recursive_extract and not args.no_recursive_extract,
            )

    write_notes(dataset_dir, dataset_key, spec)
    print("Finished.")


if __name__ == "__main__":
    main()
