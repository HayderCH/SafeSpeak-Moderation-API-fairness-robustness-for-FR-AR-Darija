"""Generate similarity statistics between original and perturbed text columns.

Example:
    python -m scripts.qa.similarity_report \
        --input data/processed/synthetic/adversarial/fr/hatexplain_fr_adv_typo.csv \
        --original-column adv_original_text \
        --perturbed-column text \
        --threshold 0.65 \
        --top-k 20 \
        --output data/review/hatexplain_fr_adv_typo_similarity.csv
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute similarity metrics for perturbed datasets."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the CSV file to inspect.",
    )
    parser.add_argument(
        "--original-column",
        default="adv_original_text",
        help="Column containing the original (pre-perturbation) text.",
    )
    parser.add_argument(
        "--perturbed-column",
        default="text",
        help="Column containing the perturbed text that will be evaluated.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Similarity threshold. Rows below this value will be flagged.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help=("Number of lowest-similarity rows to include in the console " "report."),
    )
    parser.add_argument(
        "--output",
        help="Optional path to write a CSV with similarity metrics appended.",
    )
    parser.add_argument(
        "--manifest",
        help=(
            "Optional path to write a JSON summary with aggregate similarity "
            "statistics."
        ),
    )
    return parser.parse_args()


def _char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _token_jaccard(a: str, b: str) -> float:
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


@dataclass
class SimilaritySummary:
    count: int
    mean_char: float
    median_char: float
    min_char: float
    p10_char: float
    p25_char: float
    p75_char: float
    max_char: float
    mean_jaccard: float
    below_threshold: int

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean_char": self.mean_char,
            "median_char": self.median_char,
            "min_char": self.min_char,
            "p10_char": self.p10_char,
            "p25_char": self.p25_char,
            "p75_char": self.p75_char,
            "max_char": self.max_char,
            "mean_jaccard": self.mean_jaccard,
            "below_threshold": self.below_threshold,
        }


def compute_similarity(
    df: pd.DataFrame, original_column: str, perturbed_column: str
) -> pd.DataFrame:
    """Return a dataframe with similarity metrics added."""
    if original_column not in df.columns or perturbed_column not in df.columns:
        missing = {
            c for c in (original_column, perturbed_column) if c not in df.columns
        }
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    df["char_similarity"] = [
        _char_similarity(str(o or ""), str(p or ""))
        for o, p in zip(df[original_column], df[perturbed_column])
    ]
    df["token_jaccard"] = [
        _token_jaccard(str(o or ""), str(p or ""))
        for o, p in zip(df[original_column], df[perturbed_column])
    ]
    return df


def summarize_similarities(df: pd.DataFrame, threshold: float) -> SimilaritySummary:
    char_scores = df["char_similarity"].astype(float)
    token_scores = df["token_jaccard"].astype(float)
    return SimilaritySummary(
        count=len(df),
        mean_char=float(char_scores.mean()),
        median_char=float(char_scores.median()),
        min_char=float(char_scores.min()),
        p10_char=float(char_scores.quantile(0.10)),
        p25_char=float(char_scores.quantile(0.25)),
        p75_char=float(char_scores.quantile(0.75)),
        max_char=float(char_scores.max()),
        mean_jaccard=float(token_scores.mean()),
        below_threshold=int((char_scores < threshold).sum()),
    )


def format_top_rows(df: pd.DataFrame, columns: Iterable[str], k: int) -> str:
    preview_cols = [col for col in columns if col in df.columns]
    return (
        df.sort_values("char_similarity", ascending=True)
        .head(k)[preview_cols]
        .to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    df_metrics = compute_similarity(df, args.original_column, args.perturbed_column)
    summary = summarize_similarities(df_metrics, args.threshold)

    print("=== Similarity Summary ===")
    print(json.dumps(summary.to_dict(), indent=2))
    print()

    flagged = df_metrics[df_metrics["char_similarity"] < args.threshold]
    print(f"Rows below threshold ({args.threshold}): {len(flagged)}")
    if not flagged.empty:
        preview_cols = [
            args.original_column,
            args.perturbed_column,
            "char_similarity",
            "token_jaccard",
            "adv_recipes" if "adv_recipes" in df_metrics.columns else None,
            "label" if "label" in df_metrics.columns else None,
            "source_dataset" if "source_dataset" in df_metrics.columns else None,
        ]
        preview_cols = [c for c in preview_cols if c]
        print()
        print("=== Lowest Similarity Preview ===")
        print(format_top_rows(flagged, preview_cols, args.top_k))
        print()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(output_path, index=False)
        print(f"Wrote similarity metrics to {output_path}")

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_data = summary.to_dict()
        manifest_data.update(
            {
                "input": str(input_path),
                "threshold": args.threshold,
                "flagged_rows": int(summary.below_threshold),
            }
        )
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        print(f"Wrote summary manifest to {manifest_path}")


if __name__ == "__main__":
    main()
