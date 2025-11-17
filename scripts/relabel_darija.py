import argparse
import re
from pathlib import Path

import pandas as pd


def detect_darija_features(text: str) -> dict[str, bool]:
    """Detect Darija linguistic features in text."""
    if not isinstance(text, str):
        return {}

    features = {
        "mixed_script": False,
        "arabizi_words": False,
        "darija_particles": False,
        "latin_numbers": False,
        "code_switching": False,
    }

    # Check for mixed script (Arabic + Latin)
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    latin_chars = sum(1 for c in text if "a" <= c.lower() <= "z")
    features["mixed_script"] = arabic_chars > 0 and latin_chars > 0

    # Check for Arabizi patterns (Latin script mimicking Arabic)
    arabizi_patterns = [
        r"\b(ana|enta|enti|howwa|heya|ahna|ntouma)\b",  # pronouns
        r"\b(f|fi|min|ila|ala|ma3a|wla|ola)\b",  # prepositions
        r"\b(ach|chi|chou|kay|kif|ila|bezaf)\b",  # Darija words
        r"\b(wah|wahed|jouj|tlata|rb3a|khmsa)\b",  # numbers
    ]
    features["arabizi_words"] = any(
        re.search(pattern, text.lower()) for pattern in arabizi_patterns
    )

    # Check for Darija particles and connectors
    darija_particles = [
        r"\b(wla|ola|ila|bezaf|chwaya|kay|kif)\b",
        r"\b(wah|wahed|jouj|tlata|arba|khamsa)\b",
        r"\b(ana|enta|enti|ahna|ntouma|houma)\b",
    ]
    features["darija_particles"] = any(
        re.search(pattern, text.lower()) for pattern in darija_particles
    )

    # Check for Latin numbers mixed with Arabic
    features["latin_numbers"] = bool(re.search(r"[a-zA-Z]+\d+|\d+[a-zA-Z]+", text))

    # Code switching (alternating between scripts)
    arabic_segments = len(re.findall(r"[\u0600-\u06FF]+", text))
    latin_segments = len(re.findall(r"[a-zA-Z]+", text))
    features["code_switching"] = arabic_segments > 1 and latin_segments > 1

    return features


def classify_language(text: str, current_label: str) -> str:
    """Classify text language based on features."""
    if current_label == "darija":
        return "darija"

    features = detect_darija_features(text)

    # Strong indicators for Darija
    if features.get("arabizi_words", False) or features.get("code_switching", False):
        return "darija"

    # Mixed script with Darija particles
    if features.get("mixed_script", False) and (
        features.get("darija_particles", False) or features.get("latin_numbers", False)
    ):
        return "darija"

    # Keep original label if no strong Darija indicators
    return current_label


def relabel_darija_content(input_dir: Path, output_dir: Path) -> None:
    """Relabel Darija content in datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total_samples": 0, "relabeled_to_darija": 0, "datasets_processed": 0}

    for csv_file in input_dir.glob("*.csv"):
        if "en" in csv_file.name:
            continue

        print(f"Processing {csv_file.name}...")
        df = pd.read_csv(csv_file)

        original_darija = len(df[df["language"] == "darija"])
        relabeled_count = 0

        # Apply language classification
        for idx, row in df.iterrows():
            new_label = classify_language(row["text"], row["language"])
            if new_label != row["language"]:
                df.at[idx, "language"] = new_label
                relabeled_count += 1

        # Save updated dataset
        output_file = output_dir / csv_file.name
        df.to_csv(output_file, index=False)

        final_darija = len(df[df["language"] == "darija"])

        print(f"  Original Darija: {original_darija}")
        print(f"  Relabeled: {relabeled_count}")
        print(f"  Final Darija: {final_darija}")

        stats["total_samples"] += len(df)
        stats["relabeled_to_darija"] += relabeled_count
        stats["datasets_processed"] += 1

    print("\nRelabeling Summary:")
    print(f"  Total samples processed: {stats['total_samples']}")
    print(f"  Samples relabeled to Darija: {stats['relabeled_to_darija']}")
    print(f"  Datasets processed: {stats['datasets_processed']}")


def main():
    parser = argparse.ArgumentParser(
        description="Identify and relabel Darija content in Arabic datasets"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Input directory with processed CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/relabeled"),
        help="Output directory for relabeled datasets",
    )

    args = parser.parse_args()
    relabel_darija_content(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
