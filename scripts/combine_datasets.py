#!/usr/bin/env python3
"""
Combine all processed datasets into a final training dataset.
"""

import argparse
from pathlib import Path

import pandas as pd


def combine_datasets(
    relabeled_dir: Path, augmented_darija: Path, output_file: Path
) -> None:
    """Combine all datasets into final training set."""
    all_data = []

    # Load all relabeled datasets
    for csv_file in relabeled_dir.glob("*.csv"):
        print(f"Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        all_data.append(df)

    # Load augmented Darija data
    if augmented_darija.exists():
        print(f"Loading augmented Darija from {augmented_darija.name}...")
        darija_df = pd.read_csv(augmented_darija)
        all_data.append(darija_df)

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Remove duplicates based on text content
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["text"])
    final_count = len(combined_df)

    print(f"\nCombined {initial_count} total samples")
    print(f"After deduplication: {final_count} unique samples")

    # Save final dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)

    print(f"Saved final dataset to {output_file}")

    # Show final language distribution
    print("\nFinal language distribution:")
    lang_counts = combined_df["language"].value_counts()
    for lang, count in sorted(lang_counts.items()):
        pct = (count / final_count) * 100
        print(f"  {lang}: {count:,} ({pct:.1f}%)")

    # Show augmentation breakdown
    print("\nAugmentation breakdown:")
    aug_counts = combined_df["augmentation_type"].value_counts()
    for aug_type, count in aug_counts.items():
        pct = (count / final_count) * 100
        print(f"  {aug_type}: {count:,} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Combine all datasets into final training set"
    )
    parser.add_argument(
        "--relabeled-dir",
        type=Path,
        default=Path("data/relabeled"),
        help="Directory with relabeled datasets",
    )
    parser.add_argument(
        "--augmented-darija",
        type=Path,
        default=Path("data/augmented/darija_augmented.csv"),
        help="Augmented Darija dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/final/train_combined.csv"),
        help="Final combined dataset",
    )

    args = parser.parse_args()
    combine_datasets(args.relabeled_dir, args.augmented_darija, args.output)


if __name__ == "__main__":
    main()
