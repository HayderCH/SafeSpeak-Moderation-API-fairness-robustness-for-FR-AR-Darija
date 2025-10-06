"""Generate back-translated synthetic data from a canonical SafeSpeak CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from safespeak.translation import Translator, infer_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a back-translated synthetic dataset from a canonical CSV."
        ),
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
        help="Where to write the back-translated CSV",
    )
    parser.add_argument(
        "--source-language",
        type=str,
        required=True,
        help="Language code of the input text (e.g., fr)",
    )
    parser.add_argument(
        "--pivot-language",
        type=str,
        default="en",
        help="Intermediate pivot language code (default: en)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column that contains the text to augment",
    )
    parser.add_argument(
        "--language-column",
        type=str,
        default="language",
        help="Column that stores the text language (if present)",
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
        help="Optional limit on number of rows to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device override: auto (default), cpu, cuda, or cuda:<index>",
    )
    parser.add_argument(
        "--forward-model",
        type=str,
        default=None,
        help="Optional override for the source->pivot translation model",
    )
    parser.add_argument(
        "--backward-model",
        type=str,
        default=None,
        help="Optional override for the pivot->source translation model",
    )
    parser.add_argument(
        "--drop-identical",
        action="store_true",
        help="Drop rows where the back-translation equals the original text",
    )
    parser.add_argument(
        "--dedupe-by",
        type=str,
        default=None,
        help="Optional column name to deduplicate on after augmentation",
    )
    return parser.parse_args()


def resolve_model(
    override: Optional[str],
    source_lang: str,
    target_lang: str,
) -> str:
    if override:
        return override
    return infer_model(source_lang, target_lang)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    if args.text_column not in df.columns:
        raise ValueError(f"Missing text column '{args.text_column}' in input CSV")

    forward_model = resolve_model(
        args.forward_model,
        args.source_language,
        args.pivot_language,
    )
    backward_model = resolve_model(
        args.backward_model,
        args.pivot_language,
        args.source_language,
    )

    forward_translator = Translator(
        forward_model,
        batch_size=args.batch_size,
        device=args.device,
    )
    backward_translator = Translator(
        backward_model,
        batch_size=args.batch_size,
        device=args.device,
    )

    original_texts = df[args.text_column].fillna("").astype(str)
    pivot_texts = forward_translator.translate(
        original_texts,
        show_progress=True,
        progress_desc=f"{args.source_language}->{args.pivot_language}",
    )
    back_texts = backward_translator.translate(
        pivot_texts,
        show_progress=True,
        progress_desc=f"{args.pivot_language}->{args.source_language}",
    )

    if len(pivot_texts) != len(df) or len(back_texts) != len(df):
        raise RuntimeError("Translation step produced mismatched lengths")

    augmented = df.copy()
    augmented["bt_original_text"] = original_texts
    augmented["bt_pivot_text"] = pivot_texts
    augmented[args.text_column] = back_texts

    augmented[args.language_column] = args.source_language

    if "source_text" not in augmented.columns:
        augmented["source_text"] = original_texts

    augmented["augmentation_type"] = "back_translation"
    augmented["augmentation_method"] = (
        "back_translation:"
        f"{args.source_language}->{args.pivot_language}"
        f"->{args.source_language}"
    )
    augmented["augmentation_forward_model"] = forward_model
    augmented["augmentation_backward_model"] = backward_model
    augmented["augmentation_pivot_language"] = args.pivot_language

    if args.drop_identical:
        augmented = augmented[
            augmented[args.text_column] != augmented["bt_original_text"]
        ].copy()

    if args.dedupe_by and args.dedupe_by in augmented.columns:
        augmented = augmented.drop_duplicates(subset=[args.dedupe_by])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(args.output, index=False)

    message = (
        "Saved back-translated dataset to "
        f"{args.output} (rows={len(augmented)}, "
        f"pivot={args.pivot_language})."
    )
    print(message)


if __name__ == "__main__":
    main()
