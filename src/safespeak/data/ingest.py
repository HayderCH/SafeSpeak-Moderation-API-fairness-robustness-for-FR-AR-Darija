"""Utilities for ingesting raw datasets into the SafeSpeak canonical schema."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

CANONICAL_COLUMNS = [
    "id",
    "text",
    "label",
    "language",
    "split",
    "source",
]


@dataclass
class SanitizeOptions:
    drop_duplicates: bool = True
    drop_empty_text: bool = True
    strip_text: bool = True


@dataclass
class IngestConfig:
    input_path: Path
    output_path: Optional[Path] = None
    format: str = "csv"
    text_column: str = "text"
    label_column: str = "label"
    language_column: Optional[str] = None
    language: Optional[str] = None
    split_column: Optional[str] = "split"
    default_split: str = "train"
    label_mapping: Dict[str, str] | None = None
    positive_label_columns: List[str] = field(default_factory=list)
    positive_label_value: Optional[str] = None
    negative_label_value: Optional[str] = None
    passthrough_columns: List[str] = field(default_factory=list)
    static_columns: Dict[str, Any] = field(default_factory=dict)
    sanitize: SanitizeOptions = field(default_factory=SanitizeOptions)
    save_format: str = "csv"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IngestConfig:
        sanitize_cfg = data.get("sanitize", {})
        sanitize = SanitizeOptions(**sanitize_cfg)
        static_columns = data.get("static_columns", {})
        cfg = cls(
            input_path=Path(data["input_path"]),
            output_path=(
                Path(data["output_path"]) if data.get("output_path") else None
            ),
            format=data.get("format", "csv").lower(),
            text_column=data.get("text_column", "text"),
            label_column=data.get("label_column", "label"),
            language_column=data.get("language_column"),
            language=data.get("language"),
            split_column=data.get("split_column", "split"),
            default_split=data.get("default_split", "train"),
            label_mapping=data.get("label_mapping"),
            positive_label_columns=data.get("positive_label_columns", []),
            positive_label_value=data.get("positive_label_value"),
            negative_label_value=data.get("negative_label_value"),
            passthrough_columns=data.get("passthrough_columns", []),
            static_columns=static_columns,
            sanitize=sanitize,
            save_format=data.get("save_format", "parquet").lower(),
        )
        return cfg


def load_config(path: Path | str) -> IngestConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        config_dict = yaml.safe_load(fh)
    return IngestConfig.from_dict(config_dict)


def read_raw_dataset(config: IngestConfig) -> pd.DataFrame:
    if not config.input_path.exists():
        raise FileNotFoundError(config.input_path)

    if config.format == "csv":
        df = pd.read_csv(config.input_path)
    elif config.format == "tsv":
        df = pd.read_csv(config.input_path, sep="\t")
    elif config.format in {"jsonl", "json"}:
        df = pd.read_json(config.input_path, lines=config.format == "jsonl")
    elif config.format == "parquet":
        df = pd.read_parquet(config.input_path)
    elif config.format in {"xlsx", "xls", "excel"}:
        df = pd.read_excel(config.input_path)
    else:
        raise ValueError(f"Unsupported input format: {config.format}")
    return df


def sanitize_dataframe(df: pd.DataFrame, config: IngestConfig) -> pd.DataFrame:
    if config.sanitize.drop_duplicates:
        df = df.drop_duplicates()
    if config.sanitize.drop_empty_text:
        df = df[
            df[config.text_column].notna() & (df[config.text_column].str.strip() != "")
        ]
    if config.sanitize.strip_text:
        df[config.text_column] = df[config.text_column].astype(str).str.strip()
    return df.reset_index(drop=True)


def apply_label_mapping(series: pd.Series, config: IngestConfig) -> pd.Series:
    if not config.label_mapping:
        return series.astype(str)
    mapped = series.astype(str).map(config.label_mapping)
    if mapped.isna().any():
        missing = series[mapped.isna()].unique().tolist()
        raise ValueError(f"Label mapping missing for values: {missing}")
    return mapped


def _coerce_positive_flag(value: Any) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return int(float(value) > 0)
    value_str = str(value).strip().lower()
    if not value_str:
        return 0
    if value_str in {"1", "true", "yes", "y"}:
        return 1
    if value_str in {"0", "false", "no", "n"}:
        return 0
    try:
        return int(float(value_str) > 0)
    except ValueError:
        return 0


def to_canonical(df: pd.DataFrame, config: IngestConfig) -> pd.DataFrame:
    df = sanitize_dataframe(df, config)
    df_canonical = pd.DataFrame()
    df_canonical["id"] = df.get("id", pd.RangeIndex(start=1, stop=len(df) + 1))
    df_canonical["text"] = df[config.text_column].astype(str)

    if config.positive_label_columns:
        missing_cols = [
            col for col in config.positive_label_columns if col not in df.columns
        ]
        if missing_cols:
            raise ValueError(
                "Positive label columns missing from dataset: " f"{missing_cols}"
            )
        if not config.positive_label_value or not config.negative_label_value:
            raise ValueError(
                "positive_label_columns requires positive_label_value and"
                " negative_label_value"
            )
        positive_flags = df[config.positive_label_columns].apply(
            lambda col: col.map(_coerce_positive_flag)
        )
        label_source = (
            positive_flags.sum(axis=1)
            .gt(0)
            .map(
                {
                    True: config.positive_label_value,
                    False: config.negative_label_value,
                }
            )
        )
    else:
        label_source = df[config.label_column]

    df_canonical["label"] = apply_label_mapping(label_source, config)

    if config.language_column:
        df_canonical["language"] = df[config.language_column].astype(str)
    elif config.language:
        df_canonical["language"] = config.language
    else:
        df_canonical["language"] = "unknown"

    if config.split_column and config.split_column in df.columns:
        df_canonical["split"] = df[config.split_column].fillna(config.default_split)
    else:
        df_canonical["split"] = config.default_split

    df_canonical["source"] = config.static_columns.get("source", "unknown")

    for col, value in config.static_columns.items():
        if col not in df_canonical.columns:
            df_canonical[col] = value

    if config.passthrough_columns:
        missing = [col for col in config.passthrough_columns if col not in df.columns]
        if missing:
            raise ValueError("Passthrough columns missing from dataset: " f"{missing}")
        for col in config.passthrough_columns:
            df_canonical[col] = df[col]

    missing = [col for col in CANONICAL_COLUMNS if col not in df_canonical.columns]
    if missing:
        raise ValueError(f"Canonical dataset missing columns: {missing}")

    return df_canonical


def save_dataset(df: pd.DataFrame, config: IngestConfig) -> Path:
    if not config.output_path:
        raise ValueError("output_path must be specified to save dataset")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.save_format == "csv":
        df.to_csv(config.output_path, index=False)
    elif config.save_format == "jsonl":
        df.to_json(
            config.output_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )
    elif config.save_format == "parquet":
        df.to_parquet(config.output_path, index=False)
    else:
        raise ValueError(f"Unsupported save format: {config.save_format}")
    return config.output_path


def ingest_dataset(config: IngestConfig) -> pd.DataFrame:
    df_raw = read_raw_dataset(config)
    df_canonical = to_canonical(df_raw, config)
    if config.output_path:
        save_dataset(df_canonical, config)
    return df_canonical


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest raw dataset into canonical SafeSpeak schema"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to ingestion YAML configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    df = ingest_dataset(config)
    print(
        "Ingested",
        len(df),
        "rows into",
        (
            config.output_path
            if config.output_path
            else "dataframe (no output path specified)"
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
