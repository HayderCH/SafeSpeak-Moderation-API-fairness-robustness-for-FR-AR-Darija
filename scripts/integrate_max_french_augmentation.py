#!/usr/bin/env python3
"""
Integrate all French data sources for maximum augmentation.

This script combines:
1. Original French-enhanced data (23,726 samples)
2. Jigsaw translated data (18,000 samples)
3. Back-translated variations (4,000 samples)
"""

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load all French data sources
    logger.info("Loading all French data sources...")

    # Original French-enhanced data
    original_french = pd.read_csv("data/final/train_corrected_french_enhanced.csv")
    original_french = original_french[original_french["language"] == "fr"].copy()
    logger.info(f"Original French samples: {len(original_french)}")

    # Jigsaw translated data
    jigsaw_translated = pd.read_csv("data/processed/jigsaw_french_translated.csv")
    # Convert is_toxic to label format
    jigsaw_translated["label"] = jigsaw_translated["is_toxic"].map(
        {0: "Neutral", 1: "Toxic"}
    )
    jigsaw_translated = jigsaw_translated[["text", "label", "language"]].copy()
    logger.info(f"Jigsaw translated samples: {len(jigsaw_translated)}")

    # Back-translated variations
    backtranslated = pd.read_csv("data/processed/french_backtranslated_jigsaw.csv")
    backtranslated = backtranslated[["text", "label", "language"]].copy()
    logger.info(f"Back-translated samples: {len(backtranslated)}")

    # Non-French data
    non_french = pd.read_csv("data/final/train_corrected_french_enhanced.csv")
    non_french = non_french[non_french["language"] != "fr"].copy()
    logger.info(f"Non-French samples: {len(non_french)}")

    # Combine all French sources
    all_french = pd.concat(
        [original_french, jigsaw_translated, backtranslated], ignore_index=True
    )

    # Remove duplicates based on text content
    initial_count = len(all_french)
    all_french = all_french.drop_duplicates(subset=["text"])
    duplicates_removed = initial_count - len(all_french)
    logger.info(f"Removed {duplicates_removed} duplicate texts")

    # Shuffle French data
    all_french = all_french.sample(frac=1, random_state=42).reset_index(drop=True)

    # Combine with non-French data
    final_dataset = pd.concat([non_french, all_french], ignore_index=True)

    # Final shuffle
    final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save enhanced dataset
    output_path = "data/final/train_corrected_french_max_augmentation.csv"
    final_dataset.to_csv(output_path, index=False)

    logger.info(f"Saved maximally augmented dataset to {output_path}")

    # Comprehensive summary
    print("\n🎉 Maximum French Augmentation Summary:")
    print(f"Total dataset size: {len(final_dataset)}")
    print(f"French samples: {len(all_french)}")
    print(f"Non-French samples: {len(non_french)}")

    print(f"\nFrench data sources:")
    print(f"  - Original enhanced: {len(original_french)}")
    print(f"  - Jigsaw translated: {len(jigsaw_translated)}")
    print(f"  - Back-translated: {len(backtranslated)}")
    print(f"  - Total French: {len(all_french)}")

    print(f"\nLanguage distribution:")
    lang_dist = final_dataset["language"].value_counts()
    for lang, count in lang_dist.items():
        pct = count / len(final_dataset) * 100
        print(".1f")

    french_data = final_dataset[final_dataset["language"] == "fr"]
    print(f"\nFrench label distribution:")
    label_dist = french_data["label"].value_counts()
    for label, count in label_dist.items():
        pct = count / len(french_data) * 100
        print(".1f")


if __name__ == "__main__":
    main()
