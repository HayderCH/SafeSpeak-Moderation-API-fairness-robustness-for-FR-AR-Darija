from pathlib import Path
import textwrap

import pandas as pd

from safespeak.data.ingest import (
    IngestConfig,
    ingest_dataset,
    load_config,
    to_canonical,
)


def test_load_config_and_ingest(tmp_path: Path):
    raw_data = pd.DataFrame(
        {
            "body": ["Hello", "Salut", "مرحبا"],
            "label_raw": ["neutral", "toxic", "toxic"],
            "lang": ["en", "fr", "ar"],
        }
    )
    raw_path = tmp_path / "raw.csv"
    raw_data.to_csv(raw_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            input_path: raw.csv
            output_path: processed.csv
            format: csv
            text_column: body
            label_column: label_raw
            language_column: lang
            label_mapping:
              neutral: Neutral
              toxic: Toxic
            save_format: csv
            """
        )
    )

    config = load_config(config_path)
    config.input_path = raw_path
    config.output_path = tmp_path / "processed.csv"

    df = ingest_dataset(config)
    assert set(df.columns) >= {
        "id",
        "text",
        "label",
        "language",
        "split",
        "source",
    }
    assert df["label"].tolist() == ["Neutral", "Toxic", "Toxic"]
    assert df["language"].tolist() == ["en", "fr", "ar"]
    assert config.output_path.exists()


def test_to_canonical_uses_default_language_and_split():
    cfg = IngestConfig(input_path=Path("dummy"))
    df = pd.DataFrame({"text": ["Hello"], "label": ["neutral"]})
    cfg.text_column = "text"
    cfg.label_column = "label"
    cfg.static_columns = {"source": "test"}

    canonical = to_canonical(df, cfg)
    assert canonical.loc[0, "language"] == "unknown"
    assert canonical.loc[0, "split"] == "train"
    assert canonical.loc[0, "source"] == "test"
