#!/usr/bin/env python3
"""
Extract all Darija samples from relabeled datasets into a single CSV.
"""

import argparse
from pathlib import Path

import pandas as pd


def extract_darija_samples(input_dir: Path, output_file: Path) -> None:
    """Extract all Darija samples from relabeled datasets."""
    all_darija_samples = []

    for csv_file in input_dir.glob("*.csv"):
        print(f"Processing {csv_file.name}...")
        df = pd.read_csv(csv_file)

        # Filter for Darija samples
        darija_df = df[df["language"] == "darija"].copy()

        if len(darija_df) > 0:
            # Add source information
            darija_df["original_source"] = csv_file.stem
            all_darija_samples.append(darija_df)

            print(f"  Found {len(darija_df)} Darija samples")

    if not all_darija_samples:
        print("No Darija samples found!")
        return

    # Combine all Darija samples
    combined_df = pd.concat(all_darija_samples, ignore_index=True)

    # Remove duplicates based on text content
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["text"])
    final_count = len(combined_df)

    print(f"\nCombined {initial_count} Darija samples")
    print(f"After deduplication: {final_count} unique samples")

    # Save to output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)

    print(f"Saved to {output_file}")

    # Show distribution by source
    print("\nDistribution by source:")
    source_counts = combined_df["original_source"].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Darija samples for augmentation"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/relabeled"),
        help="Input directory with relabeled CSV files",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/augmented/darija_samples.csv"),
        help="Output CSV file for Darija samples",
    )

    args = parser.parse_args()
    extract_darija_samples(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
