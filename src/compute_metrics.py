#!/usr/bin/env python3
"""
Compute classification metrics from result JSON files.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute classification metrics from a result JSON file.")
    parser.add_argument("--result-json", required=True, type=Path, help="Path to a JSON array of results.")
    parser.add_argument("--output", required=True, type=Path, help="Output text file for metric summary.")
    parser.add_argument(
        "--label-field",
        default="label",
        help="Ground-truth label field in each result record.",
    )
    parser.add_argument(
        "--prediction-field",
        default="answer",
        help="Prediction label field in each result record.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional note written at the top of the metric report.",
    )
    return parser.parse_args()


def normalize_label(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def load_records(path: Path) -> List[dict]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return parsed


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_per_label_metrics(pairs: Iterable[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
    labels = sorted({label for label, _ in pairs} | {pred for _, pred in pairs})
    counts = {label: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for label in labels}

    for gold, pred in pairs:
        counts[gold]["support"] += 1
        if gold == pred:
            counts[gold]["tp"] += 1
        else:
            counts[pred]["fp"] += 1
            counts[gold]["fn"] += 1

    metrics: Dict[str, Dict[str, float]] = {}
    for label, values in counts.items():
        precision = safe_div(values["tp"], values["tp"] + values["fp"])
        recall = safe_div(values["tp"], values["tp"] + values["fn"])
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(values["support"]),
        }
    return metrics


def summarize_metrics(per_label: Dict[str, Dict[str, float]], pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    total = len(pairs)
    correct = sum(1 for gold, pred in pairs if gold == pred)
    supports = {label: values["support"] for label, values in per_label.items()}
    label_count = len(per_label)

    macro_precision = safe_div(sum(v["precision"] for v in per_label.values()), label_count)
    macro_recall = safe_div(sum(v["recall"] for v in per_label.values()), label_count)
    macro_f1 = safe_div(sum(v["f1"] for v in per_label.values()), label_count)

    weighted_precision = safe_div(
        sum(v["precision"] * supports[label] for label, v in per_label.items()),
        total,
    )
    weighted_recall = safe_div(
        sum(v["recall"] * supports[label] for label, v in per_label.items()),
        total,
    )
    weighted_f1 = safe_div(
        sum(v["f1"] * supports[label] for label, v in per_label.items()),
        total,
    )

    return {
        "num_samples": float(total),
        "accuracy": safe_div(correct, total),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def format_report(
    summary: Dict[str, float],
    per_label: Dict[str, Dict[str, float]],
    label_distribution: Counter,
    note: str,
) -> str:
    lines: List[str] = []
    if note.strip():
        lines.append(note.strip())
        lines.append("")

    lines.append("Overall Metrics")
    lines.append(f"num_samples: {int(summary['num_samples'])}")
    lines.append(f"accuracy: {summary['accuracy']:.6f}")
    lines.append(f"macro_precision: {summary['macro_precision']:.6f}")
    lines.append(f"macro_recall: {summary['macro_recall']:.6f}")
    lines.append(f"macro_f1: {summary['macro_f1']:.6f}")
    lines.append(f"weighted_precision: {summary['weighted_precision']:.6f}")
    lines.append(f"weighted_recall: {summary['weighted_recall']:.6f}")
    lines.append(f"weighted_f1: {summary['weighted_f1']:.6f}")
    lines.append("")
    lines.append("Label Distribution")
    for label, count in sorted(label_distribution.items()):
        lines.append(f"{label}: {count}")
    lines.append("")
    lines.append("Per-Label Metrics")
    for label, metrics in sorted(per_label.items()):
        lines.append(
            f"{label}: precision={metrics['precision']:.6f}, "
            f"recall={metrics['recall']:.6f}, "
            f"f1={metrics['f1']:.6f}, "
            f"support={int(metrics['support'])}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    records = load_records(args.result_json.resolve())
    pairs: List[Tuple[str, str]] = []
    label_distribution: Counter = Counter()
    for record in records:
        gold = normalize_label(record.get(args.label_field))
        pred = normalize_label(record.get(args.prediction_field))
        if not gold or not pred:
            continue
        pairs.append((gold, pred))
        label_distribution[gold] += 1

    if not pairs:
        raise ValueError("No valid ground-truth/prediction pairs found.")

    per_label = compute_per_label_metrics(pairs)
    summary = summarize_metrics(per_label, pairs)
    report = format_report(summary, per_label, label_distribution, args.note)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Metric report written to: {output_path}")


if __name__ == "__main__":
    main()
