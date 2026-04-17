#!/usr/bin/env python3
"""Create plots to interpret baseline vs fine-tuned benchmark results.

This script reads:
- benchmark_results.json
- metrics_report.json (optional; used for convenience)

It outputs PNG charts under a target directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Allow importing sibling scripts when executed as a file.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.generate_metrics import CLASS_VOCAB, _tokenize_labels


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_jaccard(y_true: set[str], y_pred: set[str]) -> float:
    inter = len(y_true & y_pred)
    union = len(y_true | y_pred)
    return _safe_div(inter, union)


def _sample_f1(y_true: set[str], y_pred: set[str]) -> float:
    inter = len(y_true & y_pred)
    p = _safe_div(inter, len(y_pred))
    r = _safe_div(inter, len(y_true))
    return _safe_div(2 * p * r, p + r)


def _compute_per_class_recall(
    targets: list[set[str]],
    preds: list[set[str]],
    labels: list[str],
) -> tuple[dict[str, int], dict[str, float]]:
    support = {lbl: 0 for lbl in labels}
    tp = {lbl: 0 for lbl in labels}

    for y, yhat in zip(targets, preds):
        for lbl in labels:
            if lbl in y:
                support[lbl] += 1
                if lbl in yhat:
                    tp[lbl] += 1

    recall = {lbl: _safe_div(tp[lbl], support[lbl]) for lbl in labels}
    return support, recall


def plot_metric_comparison(report: dict[str, Any], out_dir: Path) -> None:
    metric_names = [
        "exact_match",
        "sample_f1",
        "sample_jaccard",
        "micro_f1",
        "macro_f1",
    ]

    baseline = [report["baseline"][m] * 100 for m in metric_names]
    finetuned = [report["finetuned"][m] * 100 for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.36

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width / 2, baseline, width, label="Baseline", color="#8d99ae")
    plt.bar(x + width / 2, finetuned, width, label="Fine-tuned", color="#2a9d8f")
    plt.xticks(x, [m.replace("_", " ").title() for m in metric_names], rotation=15, ha="right")
    plt.ylabel("Score (%)")
    plt.title("Satellite Scene Classification Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_comparison.png", dpi=160)
    plt.close()


def plot_improvement_bars(report: dict[str, Any], out_dir: Path) -> None:
    metric_names = [
        "exact_match",
        "sample_precision",
        "sample_recall",
        "sample_f1",
        "sample_jaccard",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_f1",
    ]

    deltas = [report["improvement"][m]["absolute"] * 100 for m in metric_names]
    y = np.arange(len(metric_names))

    plt.figure(figsize=(10, 6.5))
    colors = ["#2a9d8f" if d >= 0 else "#e76f51" for d in deltas]
    plt.barh(y, deltas, color=colors)
    plt.yticks(y, [m.replace("_", " ").title() for m in metric_names])
    plt.xlabel("Absolute Improvement (percentage points)")
    plt.title("Fine-tuned vs Baseline Improvement")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "improvement_absolute.png", dpi=160)
    plt.close()


def plot_jaccard_distribution(
    targets: list[set[str]],
    base_preds: list[set[str]],
    tuned_preds: list[set[str]],
    out_dir: Path,
) -> None:
    base_j = [_sample_jaccard(y, p) for y, p in zip(targets, base_preds)]
    tuned_j = [_sample_jaccard(y, p) for y, p in zip(targets, tuned_preds)]

    bins = np.linspace(0.0, 1.0, 21)
    plt.figure(figsize=(10, 5.5))
    plt.hist(base_j, bins=bins, alpha=0.55, label="Baseline", color="#8d99ae")
    plt.hist(tuned_j, bins=bins, alpha=0.55, label="Fine-tuned", color="#2a9d8f")
    plt.xlabel("Per-sample Jaccard")
    plt.ylabel("Count")
    plt.title("Distribution of Per-sample Jaccard Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "jaccard_distribution.png", dpi=160)
    plt.close()


def plot_per_class_recall(
    targets: list[set[str]],
    base_preds: list[set[str]],
    tuned_preds: list[set[str]],
    out_dir: Path,
    top_k: int,
) -> None:
    support, base_recall = _compute_per_class_recall(targets, base_preds, CLASS_VOCAB)
    _, tuned_recall = _compute_per_class_recall(targets, tuned_preds, CLASS_VOCAB)

    labels_sorted = sorted(CLASS_VOCAB, key=lambda c: support[c], reverse=True)[:top_k]
    y = np.arange(len(labels_sorted))
    width = 0.36

    bvals = [base_recall[l] * 100 for l in labels_sorted]
    tvals = [tuned_recall[l] * 100 for l in labels_sorted]

    plt.figure(figsize=(11, max(6, top_k * 0.35)))
    plt.barh(y - width / 2, bvals, height=width, label="Baseline", color="#8d99ae")
    plt.barh(y + width / 2, tvals, height=width, label="Fine-tuned", color="#2a9d8f")
    plt.yticks(y, labels_sorted)
    plt.xlabel("Recall (%)")
    plt.title(f"Per-class Recall (Top {top_k} Most Frequent Ground-truth Classes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_recall_topk.png", dpi=160)
    plt.close()


def plot_win_tie_loss(
    targets: list[set[str]],
    base_preds: list[set[str]],
    tuned_preds: list[set[str]],
    out_dir: Path,
) -> None:
    base_f1 = [_sample_f1(y, p) for y, p in zip(targets, base_preds)]
    tuned_f1 = [_sample_f1(y, p) for y, p in zip(targets, tuned_preds)]

    win = sum(1 for b, t in zip(base_f1, tuned_f1) if t > b)
    tie = sum(1 for b, t in zip(base_f1, tuned_f1) if t == b)
    loss = sum(1 for b, t in zip(base_f1, tuned_f1) if t < b)

    plt.figure(figsize=(6.8, 5.2))
    values = [win, tie, loss]
    labels = ["Fine-tuned better", "Tie", "Baseline better"]
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    plt.bar(labels, values, color=colors)
    plt.ylabel("Number of Samples")
    plt.title("Per-sample F1 Outcome")
    plt.tight_layout()
    plt.savefig(out_dir / "win_tie_loss.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual plots for benchmark results.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmark_results.json"),
        help="Path to benchmark results JSON.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("metrics_report.json"),
        help="Path to metrics report JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="How many most frequent classes to include in per-class recall plot.",
    )
    args = parser.parse_args()

    rows = _load_json(args.benchmark)
    report = _load_json(args.metrics)

    targets = [_tokenize_labels(row.get("ground_truth", ""), CLASS_VOCAB) for row in rows]
    base_preds = [_tokenize_labels(row.get("base_model_output", ""), CLASS_VOCAB) for row in rows]
    tuned_preds = [_tokenize_labels(row.get("finetuned_model_output", ""), CLASS_VOCAB) for row in rows]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_comparison(report, args.out_dir)
    plot_improvement_bars(report, args.out_dir)
    plot_jaccard_distribution(targets, base_preds, tuned_preds, args.out_dir)
    plot_per_class_recall(targets, base_preds, tuned_preds, args.out_dir, top_k=args.top_k)
    plot_win_tie_loss(targets, base_preds, tuned_preds, args.out_dir)

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
