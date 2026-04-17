#!/usr/bin/env python3
"""Generate baseline vs fine-tuned metrics for satellite scene classification.

Expected benchmark schema per row:
  - ground_truth: str
  - base_model_output: str
  - finetuned_model_output: str

The task is treated as multi-label classification over CORINE-style classes.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


# Class vocabulary used in prompts. Keeping an explicit list allows robust
# extraction even for class names that include commas.
CLASS_VOCAB = [
	"coniferous forest",
	"annual crops associated with permanent crops",
	"fruit trees and berry plantations",
	"water bodies",
	"inland marshes",
	"non-irrigated arable land",
	"port areas",
	"dump sites",
	"complex cultivation patterns",
	"sclerophyllous vegetation",
	"discontinuous urban fabric",
	"water courses",
	"sport and leisure facilities",
	"coastal lagoons",
	"green urban areas",
	"burnt areas",
	"peatbogs",
	"bare rock",
	"sparsely vegetated areas",
	"olive groves",
	"vineyards",
	"moors and heathland",
	"transitional woodland/shrub",
	"sea and ocean",
	"airports",
	"mixed forest",
	"industrial or commercial units",
	"agro-forestry areas",
	"permanently irrigated land",
	"rice fields",
	"broad-leaved forest",
	"beaches, dunes, sands",
	"mineral extraction sites",
	"continuous urban fabric",
	"construction sites",
	"intertidal flats",
	"natural grassland",
	"estuaries",
	"pastures",
	"road and rail networks and associated land",
	"land principally occupied by agriculture, with significant areas of natural vegetation",
	"salines",
	"salt marshes",
]


ALIASES = {
	"road and rail networks": "road and rail networks and associated land",
	"road and rail networks and associated": "road and rail networks and associated land",
	"land principally occupied by agriculture": "land principally occupied by agriculture, with significant areas of natural vegetation",
	"beaches dunes sands": "beaches, dunes, sands",
	"beaches, dunes and sands": "beaches, dunes, sands",
}


def _normalize_text(text: str) -> str:
	text = text.lower().strip()
	text = text.replace("\n", " ")
	text = re.sub(r"[^a-z0-9,\-/ ]+", " ", text)
	text = re.sub(r"\s+", " ", text)
	return text.strip(" .,")


def _tokenize_labels(raw: str, known_labels: Iterable[str]) -> set[str]:
	norm = _normalize_text(raw)
	if not norm:
		return set()

	normalized_vocab = {
		_normalize_text(lbl): lbl for lbl in known_labels
	}

	# Apply common aliases first.
	for alias, canonical in ALIASES.items():
		alias_norm = _normalize_text(alias)
		canonical_norm = _normalize_text(canonical)
		if re.search(rf"(^|[ ,]){re.escape(alias_norm)}([ ,]|$)", norm):
			norm = re.sub(
				rf"(^|[ ,]){re.escape(alias_norm)}([ ,]|$)",
				rf"\1{canonical_norm}\2",
				norm,
			)

	found: set[str] = set()
	# Longest labels first prevents partial matches from shadowing full labels.
	for norm_lbl in sorted(normalized_vocab.keys(), key=len, reverse=True):
		pattern = rf"(^|[ ,]){re.escape(norm_lbl)}([ ,]|$)"
		if re.search(pattern, norm):
			found.add(normalized_vocab[norm_lbl])

	# Conservative fallback if nothing matched: split on commas and keep raw parts.
	if not found:
		for part in [p.strip() for p in norm.split(",") if p.strip()]:
			if part in normalized_vocab:
				found.add(normalized_vocab[part])

	return found


@dataclass
class MetricBundle:
	samples: int
	exact_match: float
	sample_precision: float
	sample_recall: float
	sample_f1: float
	sample_jaccard: float
	micro_precision: float
	micro_recall: float
	micro_f1: float
	macro_f1: float


def _compute_metrics(targets: list[set[str]], preds: list[set[str]], labels: list[str]) -> MetricBundle:
	n = len(targets)
	if n == 0:
		return MetricBundle(
			samples=0,
			exact_match=0.0,
			sample_precision=0.0,
			sample_recall=0.0,
			sample_f1=0.0,
			sample_jaccard=0.0,
			micro_precision=0.0,
			micro_recall=0.0,
			micro_f1=0.0,
			macro_f1=0.0,
		)

	mlb = MultiLabelBinarizer(classes=labels)
	y_true = mlb.fit_transform([sorted(t) for t in targets])
	y_pred = mlb.transform([sorted(p) for p in preds])

	return MetricBundle(
		samples=n,
		exact_match=float(accuracy_score(y_true, y_pred)),
		sample_precision=float(precision_score(y_true, y_pred, average="samples", zero_division=0)),
		sample_recall=float(recall_score(y_true, y_pred, average="samples", zero_division=0)),
		sample_f1=float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
		sample_jaccard=float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
		micro_precision=float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
		micro_recall=float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
		micro_f1=float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
		macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
	)


def _pct_improvement(base: float, tuned: float) -> float:
	if base == 0.0:
		return 0.0 if tuned == 0.0 else float("inf")
	return ((tuned - base) / base) * 100.0


def _as_percent(x: float) -> str:
	if x == float("inf"):
		return "inf"
	return f"{x * 100:.2f}%"


def _load_rows(path: Path) -> list[dict]:
	rows = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(rows, list):
		raise ValueError("benchmark file must contain a JSON list")
	return rows


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compute baseline vs fine-tuned scene classification metrics from benchmark JSON."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("benchmark_results.json"),
		help="Path to benchmark JSON file.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("metrics_report.json"),
		help="Where to write computed metrics JSON.",
	)
	args = parser.parse_args()

	rows = _load_rows(args.input)

	required = {"ground_truth", "base_model_output", "finetuned_model_output"}
	for i, row in enumerate(rows):
		missing = [k for k in required if k not in row]
		if missing:
			raise ValueError(f"Row index {i} missing keys: {missing}")

	targets = [_tokenize_labels(row["ground_truth"], CLASS_VOCAB) for row in rows]
	base_preds = [_tokenize_labels(row["base_model_output"], CLASS_VOCAB) for row in rows]
	tuned_preds = [_tokenize_labels(row["finetuned_model_output"], CLASS_VOCAB) for row in rows]

	base = _compute_metrics(targets, base_preds, CLASS_VOCAB)
	tuned = _compute_metrics(targets, tuned_preds, CLASS_VOCAB)

	deltas = {}
	for key in asdict(base):
		if key == "samples":
			continue
		b = getattr(base, key)
		t = getattr(tuned, key)
		deltas[key] = {
			"absolute": t - b,
			"relative_percent": _pct_improvement(b, t),
		}

	report = {
		"input_file": str(args.input),
		"num_samples": len(rows),
		"task": "satellite_scene_classification_multilabel",
		"baseline": asdict(base),
		"finetuned": asdict(tuned),
		"improvement": deltas,
	}

	args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

	print(f"Samples: {len(rows)}")
	print(f"Report saved to: {args.output}")
	print("")
	print("Metric                  Baseline   Finetuned  Delta(abs)  Delta(rel)")
	print("-" * 72)
	for key in [
		"exact_match",
		"sample_precision",
		"sample_recall",
		"sample_f1",
		"sample_jaccard",
		"micro_precision",
		"micro_recall",
		"micro_f1",
		"macro_f1",
	]:
		b = getattr(base, key)
		t = getattr(tuned, key)
		d = deltas[key]["absolute"]
		rel = deltas[key]["relative_percent"]
		print(f"{key:<22}{_as_percent(b):>10}  {_as_percent(t):>10}  {_as_percent(d):>10}  {rel if rel == float('inf') else f'{rel:.2f}%':>9}")


if __name__ == "__main__":
	main()
