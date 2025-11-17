#!/usr/bin/env python3
"""
Quick test of Jigsaw to French translation.

Translates a small sample (100 texts) to test the pipeline before full run.
"""

import pandas as pd
from transformers import pipeline
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Test with small sample
    JIGSAW_PATH = (
        "data/raw/public/jigsaw-toxic-comment-classification-challenge/train.csv"
    )
    OUTPUT_PATH = "data/processed/jigsaw_french_test.csv"

    # Load and sample data
    df = pd.read_csv(JIGSAW_PATH)
    toxic_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    df["is_toxic"] = (df[toxic_cols].sum(axis=1) > 0).astype(int)

    # Sample 50 clean, 50 toxic
    clean_sample = df[df["is_toxic"] == 0].sample(50, random_state=42)
    toxic_sample = df[df["is_toxic"] == 1].sample(50, random_state=42)
    sample_df = pd.concat([clean_sample, toxic_sample]).sample(frac=1, random_state=42)

    logger.info(f"Testing with {len(sample_df)} samples")

    # Load translator
    translator = pipeline(
        "translation_en_to_fr",
        model="Helsinki-NLP/opus-mt-en-fr",
        device=0 if torch.cuda.is_available() else -1,
    )

    # Translate
    french_texts = []
    for i, text in enumerate(sample_df["comment_text"]):
        try:
            # Truncate very long texts to avoid model limits
            if len(text) > 1000:
                text = text[:1000] + "..."
            result = translator(text, max_length=512)[0]["translation_text"]
            french_texts.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"Translated {i + 1}/{len(sample_df)} texts")
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            french_texts.append(text)

    # Save results
    result_df = pd.DataFrame(
        {
            "text": french_texts,
            "original_text": sample_df["comment_text"].tolist(),
            "is_toxic": sample_df["is_toxic"].tolist(),
            "language": "fr",
        }
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved test translations to {OUTPUT_PATH}")

    # Show sample
    print("\nSample translations:")
    for i in range(min(5, len(result_df))):
        row = result_df.iloc[i]
        print(f"EN: {row['original_text'][:80]}...")
        print(f"FR: {row['text'][:80]}...")
        print(f"Toxic: {row['is_toxic']}")
        print("---")


if __name__ == "__main__":
    main()
