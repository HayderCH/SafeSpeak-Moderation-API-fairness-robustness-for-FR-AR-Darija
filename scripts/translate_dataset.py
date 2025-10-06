"""Translate a canonical SafeSpeak CSV into a target language.

This utility reads a CSV produced by the ingestion/extraction pipeline, uses a
Hugging Face translation model (e.g., MarianMT) to translate the `text` column,
and writes a new CSV with the translated content. The original text is stored
in a `source_text` column by default.

Example usage:

```powershell
python scripts/translate_dataset.py ^
  --input data/processed/hatexplain_en.csv ^
  --output data/interim/hatexplain/hatexplain_fr.csv ^
  --source-language en ^
  --target-language fr
```

Install the optional translation dependencies first:

```powershell
pip install -e .[translation]
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from safespeak.translation import Translator, infer_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate SafeSpeak canonical CSV to a target language."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--source-language",
        type=str,
        default="en",
        help="Source language code (default: en)",
    )
    parser.add_argument(
        "--target-language",
        type=str,
        default="fr",
        help="Target language code (default: fr)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column containing text",
    )
    parser.add_argument(
        "--language-column",
        type=str,
        default="language",
        help="Column indicating the language",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Translation batch size",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of rows to translate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and show dataset info without translating",
    )
    parser.add_argument(
        "--keep-source-column",
        action="store_true",
        help="If set, retain existing source_text column without overwriting",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=("Device override: auto (default), cpu, cuda, or cuda:<index>."),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    if args.text_column not in df.columns:
        raise ValueError(f"Missing text column '{args.text_column}' in input CSV")

    model_name = infer_model(args.source_language, args.target_language)

    if args.dry_run:
        print("Dry run summary:")
        print(f"  Rows available: {len(df)}")
        print(f"  Using model: {model_name}")
        print(f"  Text column: {args.text_column}")
        print(f"  Language column: {args.language_column}")
        return

    texts = df[args.text_column].fillna("").astype(str)
    translator = Translator(
        model_name,
        batch_size=args.batch_size,
        device=args.device,
    )
    translations = translator.translate(
        texts,
        show_progress=True,
        progress_desc=f"{args.source_language}->{args.target_language}",
    )

    if len(translations) != len(df):
        raise RuntimeError("Translation count mismatch")

    if not args.keep_source_column or "source_text" not in df:
        df["source_text"] = df[args.text_column]

    df[args.text_column] = translations
    if args.language_column in df.columns:
        df[args.language_column] = args.target_language
    else:
        df[args.language_column] = args.target_language

    df["translation_model"] = model_name

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(
        "Saved translated dataset to"
        f" {args.output} ({len(df)} rows, target={args.target_language})."
    )


if __name__ == "__main__":
    main()
