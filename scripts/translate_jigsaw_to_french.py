#!/usr/bin/env python3
"""
Translate Jigsaw samples to French for data augmentation.

This script samples balanced toxic/non-toxic examples from the Jigsaw dataset
and translates them to French to improve French language representation.
"""

import pandas as pd
from transformers import pipeline
from datasets import Dataset
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sample_jigsaw_for_translation(
    jigsaw_path: str, n_clean: int = 12000, n_toxic: int = 6000, random_state: int = 42
):
    """Sample balanced examples from Jigsaw dataset for translation."""
    logger.info("Loading Jigsaw dataset...")
    df = pd.read_csv(jigsaw_path)

    # Create binary toxic label
    toxic_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    df["is_toxic"] = (df[toxic_cols].sum(axis=1) > 0).astype(int)

    logger.info(
        f"Jigsaw dataset: {len(df)} samples, {df['is_toxic'].mean():.3f} toxic ratio"
    )

    # Sample balanced data
    clean_samples = df[df["is_toxic"] == 0].sample(n=n_clean, random_state=random_state)
    toxic_samples = df[df["is_toxic"] == 1].sample(n=n_toxic, random_state=random_state)

    candidates = pd.concat([clean_samples, toxic_samples])
    candidates = candidates.sample(frac=1, random_state=random_state)  # Shuffle

    logger.info(f"Selected {len(candidates)} samples for translation")
    logger.info(
        f"Translation candidates - Toxic: {candidates['is_toxic'].sum()}, Clean: {len(candidates) - candidates['is_toxic'].sum()}"
    )

    return candidates[["comment_text", "is_toxic"]]


def translate_with_dataset(texts, translator):
    """Translate texts using HuggingFace Dataset for efficient batch processing."""
    # Truncate long texts
    truncated_texts = []
    for text in texts:
        if len(text) > 800:
            text = text[:800] + "..."
        truncated_texts.append(text)

    # Create dataset
    dataset = Dataset.from_dict({"text": truncated_texts})

    # Translate using map (much faster than loop)
    def translate_batch(batch):
        try:
            results = translator(batch["text"], max_length=512)
            return {"translation": [r["translation_text"] for r in results]}
        except Exception as e:
            logger.warning(f"Batch translation failed: {e}")
            return {"translation": batch["text"]}  # Return originals on failure

    # Process in batches
    translated_dataset = dataset.map(
        translate_batch, batched=True, batch_size=32  # Larger batch for efficiency
    )

    return translated_dataset["translation"]


def main():
    # Configuration
    JIGSAW_PATH = (
        "data/raw/public/jigsaw-toxic-comment-classification-challenge/train.csv"
    )
    OUTPUT_PATH = "data/processed/jigsaw_french_translated.csv"
    N_CLEAN = 12000
    N_TOXIC = 6000

    # Create output directory
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Sample data for translation
    candidates = sample_jigsaw_for_translation(JIGSAW_PATH, N_CLEAN, N_TOXIC)

    # Initialize translator
    logger.info("Loading Helsinki-NLP English to French translation model...")
    translator = pipeline(
        "translation_en_to_fr",
        model="Helsinki-NLP/opus-mt-en-fr",
        device=0 if torch.cuda.is_available() else -1,
    )

    # Translate texts
    logger.info("Starting translation to French using dataset batch processing...")
    french_texts = translate_with_dataset(
        candidates["comment_text"].tolist(), translator
    )

    # Create translated dataset
    translated_df = pd.DataFrame(
        {
            "text": french_texts,
            "original_text": candidates["comment_text"].tolist(),
            "is_toxic": candidates["is_toxic"].tolist(),
            "language": "fr",
        }
    )

    # Save results
    translated_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {len(translated_df)} translated samples to {OUTPUT_PATH}")

    # Summary
    print("\nTranslation Summary:")
    print(f"Total translated: {len(translated_df)}")
    print(f"Toxic samples: {translated_df['is_toxic'].sum()}")
    print(f"Clean samples: {len(translated_df) - translated_df['is_toxic'].sum()}")
    print(f"Toxic ratio: {translated_df['is_toxic'].mean():.3f}")


if __name__ == "__main__":
    main()
