"""Comprehensive dataset analysis and visualization for SafeSpeak."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safespeak.preprocessing.normalize import normalize_text

# Optional visualization imports
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from wordcloud import WordCloud

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

VISUALIZATIONS_AVAILABLE = MATPLOTLIB_AVAILABLE and WORDCLOUD_AVAILABLE


def load_all_datasets(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all canonicalized datasets, excluding English datasets."""
    datasets = {}
    processed_dir = data_dir / "processed"

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")

    csv_files = list(processed_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {processed_dir}")

    # Exclude English datasets as per strategy decision
    english_datasets = {"hatexplain_en", "jigsaw_en"}

    for csv_file in csv_files:
        dataset_name = csv_file.stem
        if dataset_name in english_datasets:
            print(f"⏭️ Skipping English dataset: {dataset_name}")
            continue

        try:
            df = pd.read_csv(csv_file)
            datasets[dataset_name] = df
            print(f"✓ Loaded {dataset_name}: {len(df)} samples")
        except Exception as e:
            print(f"✗ Failed to load {dataset_name}: {e}")

    return datasets


def analyze_text_statistics(df: pd.DataFrame, name: str) -> dict[str, Any]:
    """Analyze text statistics for a dataset."""
    texts = df["text"].fillna("").astype(str)

    # Basic text stats
    text_lengths = texts.str.len()
    word_counts = texts.str.split().str.len()

    # Normalized text analysis
    normalized_texts = [normalize_text(text, transliterate=False) for text in texts]
    normalized_lengths = [len(text) for text in normalized_texts]
    normalized_word_counts = [len(text.split()) for text in normalized_texts]

    return {
        "dataset": name,
        "total_samples": len(df),
        "text_length_stats": {
            "mean": float(text_lengths.mean()),
            "median": float(text_lengths.median()),
            "min": int(text_lengths.min()),
            "max": int(text_lengths.max()),
            "std": float(text_lengths.std()),
        },
        "word_count_stats": {
            "mean": float(word_counts.mean()),
            "median": float(word_counts.median()),
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
            "std": float(word_counts.std()),
        },
        "normalized_length_stats": {
            "mean": float(np.mean(normalized_lengths)),
            "median": float(np.median(normalized_lengths)),
            "min": int(np.min(normalized_lengths)),
            "max": int(np.max(normalized_lengths)),
            "std": float(np.std(normalized_lengths)),
        },
        "normalized_word_stats": {
            "mean": float(np.mean(normalized_word_counts)),
            "median": float(np.median(normalized_word_counts)),
            "min": int(np.min(normalized_word_counts)),
            "max": int(np.max(normalized_word_counts)),
            "std": float(np.std(normalized_word_counts)),
        },
    }


def analyze_label_distributions(datasets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Analyze label distributions across all datasets."""
    results = {}

    for name, df in datasets.items():
        if "label" not in df.columns:
            print(f"⚠️  No 'label' column in {name}")
            continue

        label_counts = df["label"].value_counts()
        label_percentages = (label_counts / len(df) * 100).round(2)

        results[name] = {
            "total_samples": len(df),
            "unique_labels": len(label_counts),
            "label_distribution": label_counts.to_dict(),
            "label_percentages": label_percentages.to_dict(),
            "most_common_label": label_counts.index[0],
            "label_balance_score": float(
                label_counts.min() / label_counts.max()
            ),  # Balance ratio
        }

    return results


def analyze_language_distributions(datasets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Analyze language distributions across datasets."""
    results = {}

    for name, df in datasets.items():
        if "language" not in df.columns:
            results[name] = {"error": "No language column"}
            continue

        lang_counts = df["language"].value_counts()
        lang_percentages = (lang_counts / len(df) * 100).round(2)

        results[name] = {
            "total_samples": len(df),
            "unique_languages": len(lang_counts),
            "language_distribution": lang_counts.to_dict(),
            "language_percentages": lang_percentages.to_dict(),
            "primary_language": lang_counts.index[0] if len(lang_counts) > 0 else None,
        }

    return results


def analyze_split_distributions(datasets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Analyze train/dev/test splits across datasets."""
    results = {}

    for name, df in datasets.items():
        if "split" not in df.columns:
            results[name] = {"error": "No split column"}
            continue

        split_counts = df["split"].value_counts()
        split_percentages = (split_counts / len(df) * 100).round(2)

        results[name] = {
            "total_samples": len(df),
            "splits": split_counts.to_dict(),
            "split_percentages": split_percentages.to_dict(),
        }

    return results


def generate_word_clouds(datasets: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Generate word clouds for each dataset with proper Arabic support
    and stopword filtering."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define stopwords for different languages
    french_stopwords = {
        "le",
        "la",
        "les",
        "de",
        "du",
        "des",
        "et",
        "à",
        "un",
        "une",
        "dans",
        "sur",
        "pour",
        "par",
        "avec",
        "sans",
        "son",
        "sa",
        "ses",
        "ce",
        "cette",
        "ces",
        "il",
        "elle",
        "ils",
        "elles",
        "je",
        "tu",
        "nous",
        "vous",
        "me",
        "te",
        "se",
        "qui",
        "que",
        "quoi",
        "dont",
        "où",
        "quand",
        "comment",
        "pourquoi",
        "si",
        "mais",
        "ou",
        "donc",
        "or",
        "ni",
        "car",
        "comme",
        "lorsque",
        "puisque",
        "quoique",
        "bien",
    }

    # Standard Arabic stopwords (MSA - Modern Standard Arabic)
    # Note: Not used anymore since all Arabic datasets are actually Darija

    # Arabizi/Darija stopwords (Latin script - Moroccan/Tunisian dialects)
    arabizi_darija_stopwords = {
        "wa",
        "ou",
        "w",
        "o",
        "el",
        "il",
        "li",
        "fi",
        "min",
        "ala",
        "ila",
        "aan",
        "ma",
        "shi",
        "hada",
        "had",
        "hadu",
        "hadi",
        "hadik",
        "houa",
        "hia",
        "houma",
        "ana",
        "nta",
        "nti",
        "ntouma",
        "ahna",
        "dak",
        "dik",
        "houla",
        "oulak",
        "lli",
        "ill",
        "kan",
        "kant",
        "kanu",
        "kin",
        "ykoun",
        "tkoun",
        "ykounu",
        "tkounu",
        "kount",
        "kounti",
        "kountou",
        "akoun",
        "tkouni",
        "aw",
        "lakin",
        "la",
        "lan",
        "ida",
        "idan",
        "ama",
        "ima",
        "thoum",
        "bal",
        "hata",
        "am",
        "in",
        "lit",
        "laal",
        "asa",
        "kaan",
        "ki",
        "f",
        "af",
        "ya",
        "yaani",
        "wallah",
        "inshallah",
        "bisslama",
        "labas",
        "chokran",
        "afak",
        "sah",
        "oui",
        "non",
        "si",
        "mais",
        "et",
        "avec",
        "pour",
        "dans",
        "sur",
        "par",
        "de",
        "du",
        "des",
    }

    for name, df in datasets.items():
        texts = df["text"].fillna("").astype(str)

        # Combine all text
        all_text = " ".join(texts)

        # Determine language and stopwords based on dataset name
        stopwords = set()

        if "fr" in name.lower() or "french" in name.lower():
            stopwords = french_stopwords
        elif any(
            keyword in name.lower()
            for keyword in ["ar", "dz", "darija", "arabizi", "tunisian", "arabic"]
        ):
            # ALL Arabic datasets in this project are Darija/Arabizi, not MSA Arabic
            stopwords = arabizi_darija_stopwords

        # Filter out stopwords
        words = all_text.split()
        filtered_words = [
            word for word in words if word.lower() not in stopwords and len(word) > 1
        ]
        filtered_text = " ".join(filtered_words)

        processed_text = filtered_text
        arabic_font_path = None

        try:
            from arabic_reshaper import arabic_reshaper
            from bidi.algorithm import get_display

            if re.search(r"[\u0600-\u06FF]", filtered_text):
                # Configure Arabic reshaper with ligature support
                reshaper = arabic_reshaper.ArabicReshaper(
                    {
                        "delete_harakat": False,
                        "support_ligatures": True,
                    }
                )

                # Always prefer the Amiri font when Arabic text is present
                font_path = Path(__file__).parent.parent / "fonts" / "Amiri-Regular.ttf"
                if font_path.exists():
                    arabic_font_path = str(font_path)

                try:
                    reshaped_text = reshaper.reshape(filtered_text)
                    processed_text = get_display(
                        reshaped_text,
                        base_dir="R",
                        upper_is_rtl=True,
                    )
                except (IndexError, ValueError) as bidi_error:
                    # Retry after padding neutral tokens that can break bidi
                    safe_text = re.sub(
                        r"(@\S+)",
                        lambda match: f" {match.group(1)} ",
                        filtered_text,
                    )
                    try:
                        reshaped_text = reshaper.reshape(safe_text)
                        processed_text = get_display(
                            reshaped_text,
                            base_dir="R",
                            upper_is_rtl=True,
                        )
                    except (IndexError, ValueError):
                        try:
                            reshaped_text = reshaper.reshape(filtered_text)
                            processed_text = reshaped_text[::-1]
                        except Exception:
                            processed_text = filtered_text
                        print(
                            "Warning: Arabic bidi fallback used for text: "
                            f"{filtered_text[:50]}... (error: {bidi_error})"
                        )
                except Exception as reshape_error:
                    print(
                        "Warning: Arabic reshaping failed for text: "
                        f"{filtered_text[:50]}... (error: {reshape_error})"
                    )
        except ImportError:
            # If libraries not available, leave text unmodified
            pass

        # Generate word cloud with Arabic support
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=100,
            collocations=False,
            # Additional Arabic-specific settings
            regexp=r"\w[\w']+",  # Better word tokenization
            font_path=arabic_font_path if arabic_font_path else None,
        ).generate(processed_text)

        # Save word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud - {name.replace('_', ' ').title()}")
        plt.savefig(output_dir / f"{name}_wordcloud.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Generated word cloud for {name}")


def create_summary_visualizations(
    datasets: dict[str, pd.DataFrame],
    text_stats: dict[str, Any],
    label_stats: dict[str, Any],
    output_dir: Path,
) -> None:
    """Create summary visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset sizes comparison
    dataset_sizes = {name: len(df) for name, df in datasets.items()}

    plt.figure(figsize=(12, 6))
    plt.bar(dataset_sizes.keys(), dataset_sizes.values())
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of Samples")
    plt.title("Dataset Sizes Comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Text length distributions
    plt.figure(figsize=(12, 6))
    for name, stats in text_stats.items():
        plt.hist(
            [stats["text_length_stats"]["mean"]],
            alpha=0.7,
            label=name,
            bins=20,
        )
    plt.xlabel("Average Text Length (characters)")
    plt.ylabel("Frequency")
    plt.title("Text Length Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "text_lengths.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Label balance heatmap
    balance_data = []
    dataset_names = []
    for name, stats in label_stats.items():
        if "label_balance_score" in stats:
            balance_data.append(stats["label_balance_score"])
            dataset_names.append(name)

    if balance_data:
        plt.figure(figsize=(8, 6))
        plt.barh(dataset_names, balance_data)
        plt.xlabel("Label Balance Score (min/max ratio)")
        plt.title("Dataset Label Balance")
        plt.tight_layout()
        plt.savefig(output_dir / "label_balance.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"✓ Generated summary visualizations in {output_dir}")


def generate_comprehensive_report(
    datasets: dict[str, pd.DataFrame],
    text_stats: dict[str, Any],
    label_stats: dict[str, Any],
    language_stats: dict[str, Any],
    split_stats: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate a comprehensive analysis report."""
    report = {
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "total_datasets": len(datasets),
        "total_samples": sum(len(df) for df in datasets.values()),
        "datasets": {},
        "summary": {},
    }

    # Dataset details
    for name in datasets.keys():
        report["datasets"][name] = {
            "samples": len(datasets[name]),
            "text_stats": text_stats.get(name, {}),
            "label_stats": label_stats.get(name, {}),
            "language_stats": language_stats.get(name, {}),
            "split_stats": split_stats.get(name, {}),
        }

    # Overall summary
    all_labels = []
    all_languages = []
    all_text_lengths = []

    for df in datasets.values():
        if "label" in df.columns:
            all_labels.extend(df["label"].dropna().tolist())
        if "language" in df.columns:
            all_languages.extend(df["language"].dropna().tolist())
        if "text" in df.columns:
            all_text_lengths.extend(
                df["text"].fillna("").astype(str).str.len().tolist()
            )

    report["summary"] = {
        "total_samples": sum(len(df) for df in datasets.values()),
        "unique_labels": len(set(all_labels)) if all_labels else 0,
        "unique_languages": len(set(all_languages)) if all_languages else 0,
        "label_distribution": dict(Counter(all_labels)) if all_labels else {},
        "language_distribution": dict(Counter(all_languages)) if all_languages else {},
        "text_length_stats": {
            "mean": float(np.mean(all_text_lengths)) if all_text_lengths else 0,
            "median": float(np.median(all_text_lengths)) if all_text_lengths else 0,
            "min": int(np.min(all_text_lengths)) if all_text_lengths else 0,
            "max": int(np.max(all_text_lengths)) if all_text_lengths else 0,
        },
    }

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "analysis_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        f"✓ Generated comprehensive analysis report: {output_dir / 'analysis_report.json'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive dataset analysis for SafeSpeak"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory containing processed datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis"),
        help="Output directory for analysis results and visualizations",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualizations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("🔍 SafeSpeak Dataset Analysis")
    print("=" * 50)

    # Load all datasets
    print("\n📂 Loading datasets...")
    try:
        datasets = load_all_datasets(args.data_dir)
    except Exception as e:
        print(f"❌ Failed to load datasets: {e}")
        return

    if not datasets:
        print("❌ No datasets found!")
        return

    print(
        f"✅ Loaded {len(datasets)} datasets with {sum(len(df) for df in datasets.values())} total samples"
    )

    # Analyze text statistics
    print("\n📊 Analyzing text statistics...")
    text_stats = {}
    for name, df in datasets.items():
        text_stats[name] = analyze_text_statistics(df, name)

    # Analyze label distributions
    print("🏷️  Analyzing label distributions...")
    label_stats = analyze_label_distributions(datasets)

    # Analyze language distributions
    print("🌍 Analyzing language distributions...")
    language_stats = analyze_language_distributions(datasets)

    # Analyze split distributions
    print("📋 Analyzing train/dev/test splits...")
    split_stats = analyze_split_distributions(datasets)

    # Generate visualizations
    if not args.no_visualizations:
        print("\n🎨 Generating visualizations...")
        if WORDCLOUD_AVAILABLE:
            generate_word_clouds(datasets, args.output_dir / "wordclouds")
        else:
            print("⚠️  WordCloud library not available - skipping word clouds")
            print("   Install with: pip install wordcloud")

        if MATPLOTLIB_AVAILABLE:
            create_summary_visualizations(
                datasets, text_stats, label_stats, args.output_dir / "visualizations"
            )
        else:
            print(
                "⚠️  Matplotlib library not available - skipping summary visualizations"
            )
            print("   Install with: pip install matplotlib seaborn")

    # Generate comprehensive report
    print("\n📝 Generating comprehensive report...")
    generate_comprehensive_report(
        datasets, text_stats, label_stats, language_stats, split_stats, args.output_dir
    )

    # Print summary
    print("\n🎯 Analysis Complete!")
    print(f"📁 Results saved to: {args.output_dir}")
    print("\n📈 Key Insights:")

    total_samples = sum(len(df) for df in datasets.values())
    print(f"   • Total samples: {total_samples:,}")

    all_languages = set()
    for df in datasets.values():
        if "language" in df.columns:
            all_languages.update(df["language"].dropna().unique())
    print(f"   • Languages covered: {sorted(all_languages)}")

    # Check for imbalanced datasets
    imbalanced = []
    for name, stats in label_stats.items():
        if "label_balance_score" in stats and stats["label_balance_score"] < 0.3:
            imbalanced.append(name)
    if imbalanced:
        print(f"   • Potentially imbalanced datasets: {imbalanced}")
        print("     Consider resampling or weighted loss functions")

    print("\n🚀 Next steps:")
    print("   • Review analysis_report.json for detailed statistics")
    print("   • Check visualizations/ for dataset comparisons")
    print("   • Examine wordclouds/ for content insights")
    print("   • Consider data augmentation for underrepresented classes")


if __name__ == "__main__":
    main()
