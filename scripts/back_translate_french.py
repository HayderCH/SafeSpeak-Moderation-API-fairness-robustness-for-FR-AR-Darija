#!/usr/bin/env python3
"""
Back-translate French samples for data augmentation.

This script takes French samples and back-translates them through English
to create natural variations while preserving toxicity labels.
"""

import logging
import pandas as pd
import torch
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def back_translate_french_samples(
    french_data_path: str,
    n_samples: int = 5000,
    output_path: str = "data/processed/french_backtranslated.csv",
    random_state: int = 42,
):
    """
    Back-translate French samples through English to create variations.

    Args:
        french_data_path: Path to French data CSV
        n_samples: Number of samples to back-translate
        output_path: Where to save back-translated data
        random_state: Random seed for reproducibility
    """
    logger.info("Loading French data...")
    df = pd.read_csv(french_data_path)

    # Filter to French samples only
    french_df = df[df["language"] == "fr"].copy()
    logger.info(f"Found {len(french_df)} French samples")

    # Handle different label column names
    if "label" in french_df.columns:
        label_col = "label"
        toxic_label = "Toxic"
        neutral_label = "Neutral"
    elif "is_toxic" in french_df.columns:
        label_col = "is_toxic"
        toxic_label = 1
        neutral_label = 0
    else:
        raise ValueError("No label column found in data")

    # Sample balanced toxic/non-toxic
    toxic_samples = french_df[french_df[label_col] == toxic_label]
    neutral_samples = french_df[french_df[label_col] == neutral_label]

    # Calculate samples per class
    n_toxic = min(len(toxic_samples), n_samples // 2)
    n_neutral = min(len(neutral_samples), n_samples // 2)

    logger.info(f"Sampling {n_toxic} toxic and {n_neutral} neutral samples")

    # Sample with stratification
    toxic_sampled = toxic_samples.sample(n=n_toxic, random_state=random_state)
    neutral_sampled = neutral_samples.sample(n=n_neutral, random_state=random_state)

    backtranslate_df = pd.concat([toxic_sampled, neutral_sampled], ignore_index=True)
    backtranslate_df = backtranslate_df.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    selected_count = len(backtranslate_df)
    logger.info(f"Selected {selected_count} samples for back-translation")

    # Initialize translation pipelines
    logger.info("Loading translation models...")
    fr_to_en = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-fr-en",
        device=0 if torch.cuda.is_available() else -1,
        max_length=512,
    )

    en_to_fr = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-fr",
        device=0 if torch.cuda.is_available() else -1,
        max_length=512,
    )

    def back_translate_batch(texts, batch_size=32):
        """Back-translate a batch of texts through English."""
        back_translated = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) - 1) // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")

            try:
                # French -> English
                en_translations = fr_to_en(
                    batch, max_length=512, truncation=True, do_sample=False
                )
                en_texts = [t["translation_text"] for t in en_translations]

                # English -> French (back)
                fr_back = en_to_fr(
                    en_texts,
                    max_length=512,
                    truncation=True,
                    do_sample=True,  # Add some variation
                    temperature=0.8,
                    top_p=0.9,
                )
                fr_texts = [t["translation_text"] for t in fr_back]

                back_translated.extend(fr_texts)

            except Exception as e:
                logger.warning(f"Batch translation failed: {e}")
                # Fallback: return original texts
                back_translated.extend(batch)

        return back_translated

    # Back-translate texts
    logger.info("Starting back-translation...")
    original_texts = backtranslate_df["text"].tolist()
    back_translated_texts = back_translate_batch(original_texts)

    # Create new dataframe with back-translated texts
    backtranslated_df = backtranslate_df.copy()
    backtranslated_df["text"] = back_translated_texts
    backtranslated_df["augmentation_type"] = "backtranslation"
    backtranslated_df["original_text"] = original_texts

    # Convert labels to standard format if needed
    if label_col == "is_toxic":
        backtranslated_df["label"] = backtranslated_df["is_toxic"].map(
            {0: "Neutral", 1: "Toxic"}
        )
        # Keep original column for reference
    else:
        backtranslated_df["label"] = backtranslated_df["label"]

    # Save results
    backtranslated_df.to_csv(output_path, index=False)
    saved_count = len(backtranslated_df)
    logger.info(f"Saved {saved_count} back-translated samples to " f"{output_path}")

    # Print summary
    print("\nBack-translation Summary:")
    print(f"Original samples: {len(backtranslate_df)}")
    print(f"Back-translated: {len(backtranslated_df)}")
    print("Label distribution:")
    print(backtranslated_df["label"].value_counts())

    return backtranslated_df


def main():
    # Back-translate some of the new Jigsaw French data
    back_translate_french_samples(
        french_data_path="data/processed/jigsaw_french_translated.csv",
        n_samples=4000,  # Back-translate 4K of the 18K new samples
        output_path="data/processed/french_backtranslated_jigsaw.csv",
    )


if __name__ == "__main__":
    main()
