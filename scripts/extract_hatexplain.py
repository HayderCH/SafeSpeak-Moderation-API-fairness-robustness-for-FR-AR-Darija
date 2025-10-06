"""Convert HateXplain JSON into the SafeSpeak canonical CSV format.

This script expects the HateXplain repository to be downloaded under
`data/raw/public/hatexplain/HateXplain-master/` (default). It loads the
`dataset.json` and `post_id_divisions.json` files, resolves the majority label
per example, and writes a CSV ready for ingestion into the SafeSpeak pipeline.

Usage example:

```powershell
python scripts/extract_hatexplain.py \
    --input data/raw/public/hatexplain/HateXplain-master/Data \
    --output data/interim/hatexplain/hatexplain_en.csv
```
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

CANONICAL_LABEL_MAP = {
    "hatespeech": "Hate",
    "offensive": "Toxic",
    "normal": "Neutral",
}

SEVERITY_ORDER = ["hatespeech", "offensive", "normal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HateXplain JSON data into canonical CSV format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/public/hatexplain/HateXplain-master/Data"),
        help="Directory containing HateXplain Data folder with dataset.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/hatexplain/hatexplain_en.csv"),
        help="Output CSV path to write canonicalized data",
    )
    parser.add_argument(
        "--include-rationales",
        action="store_true",
        help=(
            "If set, include concatenated rationale spans as a JSON " "string column."
        ),
    )
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def majority_vote(labels: Iterable[str]) -> str:
    counts = Counter(label.lower() for label in labels)
    if not counts:
        raise ValueError("No labels provided for majority vote")
    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]
    if len(tied) == 1:
        return tied[0]
    # Tie-break by severity (Hate > Offensive > Normal)
    for candidate in SEVERITY_ORDER:
        if candidate in tied:
            return candidate
    # Fallback to deterministic ordering
    return sorted(tied)[0]


def gather_targets(annotators: List[dict]) -> List[str]:
    targets: set[str] = set()
    for ann in annotators:
        for target in ann.get("target", []) or []:
            targets.add(target)
    return sorted(targets)


def main() -> None:
    args = parse_args()

    data_dir: Path = args.input
    if not data_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {data_dir}")

    dataset = read_json(data_dir / "dataset.json")
    splits = read_json(data_dir / "post_id_divisions.json")

    split_lookup = {}
    for split_name, post_ids in splits.items():
        for post_id in post_ids:
            split_lookup[post_id] = split_name

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "text",
        "language",
        "raw_label",
        "label",
        "split",
        "annotator_labels",
        "targets",
    ]
    if args.include_rationales:
        fieldnames.append("rationales")

    with args.output.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for post_id, payload in dataset.items():
            tokens = payload.get("post_tokens", [])
            text = " ".join(tokens).strip()

            annotators = payload.get("annotators", [])
            annotator_labels = [ann.get("label", "").lower() for ann in annotators]
            if not annotator_labels:
                # Skip entries without labels.
                continue

            raw_majority = majority_vote(annotator_labels)
            canonical_label = CANONICAL_LABEL_MAP.get(raw_majority, "Unknown")

            split = split_lookup.get(post_id, "unspecified")
            targets = gather_targets(annotators)

            row = {
                "id": post_id,
                "text": text,
                "language": "en",
                "raw_label": raw_majority,
                "label": canonical_label,
                "split": split,
                "annotator_labels": ",".join(annotator_labels),
                "targets": ";".join(targets),
            }

            if args.include_rationales:
                row["rationales"] = json.dumps(payload.get("rationales", []))

            writer.writerow(row)

    print(f"Wrote {args.output} with {len(dataset)} entries.")


if __name__ == "__main__":
    main()
