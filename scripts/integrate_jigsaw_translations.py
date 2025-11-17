#!/usr/bin/env python3
"""
Integrate Jigsaw French translations into the training dataset.

This script combines the translated Jigsaw samples with our existing
French-enhanced training data.
"""

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load datasets
    logger.info("Loading datasets...")
    current_data = pd.read_csv("data/final/train_corrected_french_enhanced.csv")
    translations = pd.read_csv("data/processed/jigsaw_french_translated.csv")

    # Map labels: is_toxic (0/1) -> Toxic/Neutral
    translations["label"] = translations["is_toxic"].map({0: "Neutral", 1: "Toxic"})

    # Select and rename columns to match our format
    translations = translations[["text", "label", "language"]].copy()

    # Remove translations from current data (keep only original French)
    current_french_only = current_data[current_data["language"] == "fr"].copy()
    current_non_french = current_data[current_data["language"] != "fr"].copy()

    logger.info(f"Current French samples: {len(current_french_only)}")
    logger.info(f"New French translations: {len(translations)}")
    logger.info(f"Non-French samples: {len(current_non_french)}")

    # Combine datasets
    combined_french = pd.concat([current_french_only, translations], ignore_index=True)
    final_dataset = pd.concat([current_non_french, combined_french], ignore_index=True)

    # Shuffle the dataset
    final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save updated dataset
    output_path = "data/final/train_corrected_french_enhanced_jigsaw.csv"
    final_dataset.to_csv(output_path, index=False)

    logger.info(f"Saved combined dataset to {output_path}")

    # Summary
    print("\nDataset Summary:")
    print(f"Total samples: {len(final_dataset)}")
    print("Language distribution:")
    print(final_dataset["language"].value_counts())
    french_samples = len(final_dataset[final_dataset["language"] == "fr"])
    print(f"\nFrench samples: {french_samples}")
    print("Label distribution (French only):")
    french_data = final_dataset[final_dataset["language"] == "fr"]
    print(french_data["label"].value_counts())


if __name__ == "__main__":
    main()
