#!/usr/bin/env python3
"""
Relabel ALL Arabic samples as Darija based on domain expertise.
"""

import argparse
from pathlib import Path

import pandas as pd


def relabel_arabic_as_darija(input_dir: Path, output_dir: Path) -> None:
    """Relabel all Arabic samples as Darija."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total_samples": 0, "arabic_relabeled": 0, "datasets_processed": 0}

    for csv_file in input_dir.glob("*.csv"):
        print(f"Processing {csv_file.name}...")
        df = pd.read_csv(csv_file)

        # Relabel ALL Arabic samples as Darija
        arabic_mask = df["language"] == "ar"
        relabeled_count = 0

        if arabic_mask.any():
            df.loc[arabic_mask, "language"] = "darija"
            relabeled_count = arabic_mask.sum()

        # Save updated dataset
        output_file = output_dir / csv_file.name
        df.to_csv(output_file, index=False)

        print(f"  Arabic samples relabeled: {relabeled_count}")

        stats["total_samples"] += len(df)
        stats["arabic_relabeled"] += relabeled_count
        stats["datasets_processed"] += 1

    print("\nMassive Relabeling Summary:")
    print(f"  Total samples processed: {stats['total_samples']}")
    print(f"  Arabic → Darija relabeling: {stats['arabic_relabeled']}")
    print(f"  Datasets processed: {stats['datasets_processed']}")


def main():
    parser = argparse.ArgumentParser(description="Relabel all Arabic samples as Darija")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/relabeled"),
        help="Input directory with relabeled CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/darija_corrected"),
        help="Output directory for Darija-corrected datasets",
    )

    args = parser.parse_args()
    relabel_arabic_as_darija(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
