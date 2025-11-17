#!/usr/bin/env python3
"""
Simple augmentation techniques for Darija text.
"""

import argparse
import random
import re
from pathlib import Path

import pandas as pd


class DarijaAugmenter:
    """Simple augmenter for Darija text using linguistic transformations."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.seed = seed

        # Darija-specific synonym mappings (Arabizi -> alternatives)
        self.synonym_map = {
            "ana": ["ana", "anah", "anaha"],
            "enta": ["enta", "ent", "enti"],
            "howwa": ["howwa", "howa", "houa"],
            "heya": ["heya", "hya", "hia"],
            "ahna": ["ahna", "ahnih", "ihna"],
            "ntouma": ["ntouma", "ntoumah", "antouma"],
            "kay": ["kay", "kif", "kayen"],
            "bezaf": ["bezaf", "bzaf", "ktir"],
            "chwaya": ["chwaya", "chwiya", "chwia"],
            "wah": ["wah", "ouahed", "wahed"],
            "jouj": ["jouj", "juj", "doj"],
            "tlata": ["tlata", "tlat", "tlatah"],
            "arba": ["arba", "arbah", "rba"],
            "khamsa": ["khamsa", "khamsah", "khmsa"],
            "ach": ["ach", "chi", "chou"],
            "ola": ["ola", "wla", "awla"],
            "ila": ["ila", "ala", "l"],
            "f": ["f", "fi", "fel"],
            "min": ["min", "men", "mn"],
            "ma3a": ["ma3a", "m3a", "maah"],
        }

        # Character-level variations for Arabizi
        self.char_variations = {
            "a": ["a", "à", "â"],
            "e": ["e", "é", "è"],
            "o": ["o", "ô", "ò"],
            "u": ["u", "û", "ù"],
            "i": ["i", "î", "ï"],
            "c": ["c", "ç"],
            "s": ["s", "š", "ş"],
            "z": ["z", "ž"],
            "g": ["g", "ğ"],
        }

    def synonym_replacement(self, text: str) -> str:
        """Replace words with Darija synonyms."""
        words = text.split()
        augmented_words = []

        for word in words:
            word_lower = word.lower()
            if word_lower in self.synonym_map:
                synonyms = self.synonym_map[word_lower]
                # Keep original case if capitalized
                replacement = random.choice(synonyms)
                if word[0].isupper():
                    replacement = replacement.capitalize()
                augmented_words.append(replacement)
            else:
                augmented_words.append(word)

        return " ".join(augmented_words)

    def character_variation(self, text: str, variation_prob: float = 0.1) -> str:
        """Apply character-level variations to Arabizi text."""
        result = []
        for char in text:
            char_lower = char.lower()
            if char_lower in self.char_variations and random.random() < variation_prob:
                variations = self.char_variations[char_lower]
                variation = random.choice(variations)
                # Preserve case
                if char.isupper():
                    variation = variation.upper()
                result.append(variation)
            else:
                result.append(char)
        return "".join(result)

    def arabizi_normalization(self, text: str) -> str:
        """Normalize common Arabizi variations."""
        # Common Arabizi normalizations
        normalizations = {
            r"\b(ent|enti|entah)\b": "enta",
            r"\b(hwa|howa|houa|houwa)\b": "howwa",
            r"\b(hya|hia|hiya)\b": "heya",
            r"\b(ahnih|ihna|ihnih)\b": "ahna",
            r"\b(kifen|kayen|kayna)\b": "kay",
            r"\b(bzaf|ktir|ktira)\b": "bezaf",
            r"\b(chwiya|chwia|chwayah)\b": "chwaya",
            r"\b(ouahed|wahed|wahd)\b": "wah",
            r"\b(juj|doj|dooj)\b": "jouj",
            r"\b(tlat|tlatah)\b": "tlata",
            r"\b(arbah|rba|arbaah)\b": "arba",
            r"\b(khamsah|khmsa)\b": "khamsa",
            r"\b(chi|chou|achnou)\b": "ach",
            r"\b(wla|awla|awlah)\b": "ola",
            r"\b(ala|el|l)\b": "ila",
            r"\b(fi|fel|fih)\b": "f",
            r"\b(men|mn|minn)\b": "min",
            r"\b(m3a|maah|maa)\b": "ma3a",
        }

        result = text
        for pattern, replacement in normalizations.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    def augment_text(self, text: str, num_variations: int = 3) -> list[str]:
        """Generate multiple augmented versions of a text."""
        variations = []

        for _ in range(num_variations):
            augmented = text

            # Apply different augmentation techniques
            if random.random() < 0.7:  # 70% chance
                augmented = self.synonym_replacement(augmented)

            if random.random() < 0.5:  # 50% chance
                augmented = self.character_variation(augmented, 0.1)

            if random.random() < 0.3:  # 30% chance
                augmented = self.arabizi_normalization(augmented)

            # Only add if different from original
            if augmented != text:
                variations.append(augmented)

        # Remove duplicates and limit to num_variations
        variations = list(set(variations))[:num_variations]

        return variations

    def augment_dataset(
        self, df: pd.DataFrame, target_samples: int = None
    ) -> pd.DataFrame:
        """Augment a dataset to reach target sample count."""
        augmented_rows = []

        # Calculate how many augmentations we need per sample
        current_samples = len(df)
        if target_samples is None:
            target_samples = current_samples * 2  # Default: double the dataset

        samples_needed = max(0, target_samples - current_samples)
        augmentations_per_sample = max(1, samples_needed // current_samples)

        print(f"Current samples: {current_samples}")
        print(f"Target samples: {target_samples}")
        print(f"Augmentations per sample: {augmentations_per_sample}")

        for _, row in df.iterrows():
            # Keep original sample
            original_row = row.copy()
            original_row["augmentation_type"] = "original"
            original_row["augmentation_method"] = "none"
            augmented_rows.append(original_row)

            # Generate augmentations
            variations = self.augment_text(row["text"], augmentations_per_sample)

            for variation in variations:
                augmented_row = row.copy()
                augmented_row["text"] = variation
                augmented_row["augmentation_type"] = "linguistic_variation"
                augmented_row["augmentation_method"] = "darija_synonym_char_variation"
                augmented_row["bt_original_text"] = row["text"]
                augmented_rows.append(augmented_row)

        result_df = pd.DataFrame(augmented_rows)

        # Limit to target samples if specified
        if len(result_df) > target_samples:
            result_df = result_df.head(target_samples)

        return result_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Augment Darija dataset using linguistic variations"
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input CSV with Darija samples"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output CSV with augmented data"
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=None,
        help="Target number of samples (default: double current count)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} Darija samples")

    # Augment
    augmenter = DarijaAugmenter(seed=args.seed)
    augmented_df = augmenter.augment_dataset(df, args.target_samples)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(args.output, index=False)

    print(f"Saved {len(augmented_df)} augmented samples to {args.output}")

    # Show augmentation stats
    aug_types = augmented_df["augmentation_type"].value_counts()
    print("\nAugmentation breakdown:")
    for aug_type, count in aug_types.items():
        print(f"  {aug_type}: {count}")


if __name__ == "__main__":
    main()
