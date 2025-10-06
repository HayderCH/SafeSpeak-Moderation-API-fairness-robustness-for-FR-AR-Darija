"""Data loading helpers for SafeSpeak datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

CANONICAL_COLUMNS = ["id", "text", "label", "language", "split", "source"]


@dataclass
class DatasetSplit:
    """Container for dataset splits."""

    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame


def load_dataset(
    path: Path | str,
    *,
    expect_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load a dataset from CSV/JSONL with canonical formatting."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".json"}:
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    if expect_columns is None:
        expect_columns = CANONICAL_COLUMNS

    missing = [col for col in expect_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    return df


def build_splits(df: pd.DataFrame) -> DatasetSplit:
    """Split dataset using the `split` column markers."""
    if "split" not in df.columns:
        raise ValueError("Dataset requires a 'split' column with values train/dev/test")

    train = df[df["split"].str.lower() == "train"].reset_index(drop=True)
    dev = df[df["split"].str.lower().isin(["dev", "val", "validation"])].reset_index(
        drop=True
    )
    test = df[df["split"].str.lower() == "test"].reset_index(drop=True)

    if train.empty or dev.empty or test.empty:
        raise ValueError(
            "One or more splits are empty. "
            "Ensure dataset includes train/dev/test rows."
        )

    return DatasetSplit(train=train, dev=dev, test=test)
