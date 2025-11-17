"""Class balancing utilities for SafeSpeak datasets."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def load_dataset(data_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Load a specific dataset."""
    processed_dir = data_dir / "processed"
    csv_file = processed_dir / f"{dataset_name}.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {csv_file}")

    return pd.read_csv(csv_file)


def analyze_balance(df: pd.DataFrame, label_col: str = "label") -> dict:
    """Analyze class balance in a dataset."""
    label_counts = df[label_col].value_counts()
    total = len(df)

    balance_score = float(label_counts.min() / label_counts.max())

    return {
        "total_samples": total,
        "label_distribution": dict(label_counts),
        "label_percentages": {
            k: round(v / total * 100, 2) for k, v in label_counts.items()
        },
        "balance_score": balance_score,
        "is_balanced": balance_score >= 0.5,
    }


def balance_dataset_smote(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    target_ratio: float = 1.0,
) -> pd.DataFrame:
    """Balance dataset using SMOTE oversampling."""
    # Prepare features
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df[text_col].fillna(""))
    y = df[label_col]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Calculate target samples per class
    class_counts = Counter(y_encoded)
    max_count = max(class_counts.values())
    target_samples = {cls: int(max_count * target_ratio) for cls in class_counts.keys()}

    # Apply SMOTE
    smote = SMOTE(sampling_strategy=target_samples, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

    # Decode labels back
    y_decoded = label_encoder.inverse_transform(y_resampled)

    # Create balanced dataframe
    balanced_df = pd.DataFrame(
        {
            text_col: [""]
            * len(y_decoded),  # Placeholder, will be filled with original texts
            label_col: y_decoded,
        }
    )

    # For SMOTE samples, we need to find similar original samples
    # This is a simplified approach - in practice, you'd want more sophisticated text generation
    balanced_texts = []

    for _, label in enumerate(y_decoded):
        # Find original texts with same label
        same_label_texts = df[df[label_col] == label][text_col].tolist()
        if same_label_texts:
            # Use original text (simplified - real implementation would generate variations)
            balanced_texts.append(np.random.choice(same_label_texts))
        else:
            balanced_texts.append("")

    balanced_df[text_col] = balanced_texts

    # Preserve other columns if they exist
    for col in df.columns:
        if col not in [text_col, label_col]:
            if col in df.columns:
                # For SMOTE samples, duplicate values from original samples
                balanced_df[col] = [
                    df[col].iloc[i % len(df)] for i in range(len(balanced_df))
                ]

    return balanced_df


def balance_dataset_undersample(
    df: pd.DataFrame, label_col: str = "label", target_ratio: float = 1.0
) -> pd.DataFrame:
    """Balance dataset using random undersampling."""
    # Get label counts
    label_counts = df[label_col].value_counts()
    min_count = label_counts.min()

    # Calculate target samples
    target_samples = int(min_count * target_ratio)

    # We need numeric labels for undersampling
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df[label_col])

    # Create sampling strategy with encoded labels
    label_to_encoded = dict(
        zip(label_encoder.classes_, range(len(label_encoder.classes_)))
    )
    target_samples_encoded = {
        label_to_encoded[label]: min(target_samples, count)
        for label, count in label_counts.items()
    }

    # Undersample majority classes
    undersampler = RandomUnderSampler(
        sampling_strategy=target_samples_encoded, random_state=42
    )

    # Get indices after undersampling
    _, y_resampled = undersampler.fit_resample(
        np.arange(len(df)).reshape(-1, 1), y_encoded
    )

    # Get corresponding rows
    balanced_df = df.iloc[y_resampled].copy()

    return balanced_df


def save_balanced_dataset(
    df: pd.DataFrame, output_dir: Path, dataset_name: str, method: str
) -> Path:
    """Save balanced dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset_name}_balanced_{method}.csv"
    df.to_csv(output_file, index=False)
    return output_file


def balance_dataset_auto(
    df: pd.DataFrame, text_col: str = "text", label_col: str = "label"
) -> tuple[pd.DataFrame, str]:
    """Automatically choose and apply balancing method based on dataset characteristics."""
    balance_info = analyze_balance(df, label_col)
    balance_score = balance_info["balance_score"]

    if balance_score >= 0.5:
        # Already reasonably balanced
        return df, "no_balancing_needed"

    # For severely imbalanced datasets, use oversampling if dataset is small
    # or undersampling if dataset is large
    total_samples = len(df)

    if total_samples < 5000:
        # Small dataset - use oversampling
        balanced_df = balance_dataset_smote(df, text_col, label_col)
        method = "smote_oversampling"
    else:
        # Large dataset - use undersampling to preserve quality
        balanced_df = balance_dataset_undersample(df, label_col)
        method = "random_undersampling"

    return balanced_df, method


def main():
    """Main balancing script."""
    parser = argparse.ArgumentParser(description="Balance SafeSpeak datasets")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Data directory path"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/balanced"),
        help="Output directory for balanced datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["narabizi_treebank", "arabizi_offensive_lang"],
        help="Datasets to balance",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "smote", "undersample"],
        default="auto",
        help="Balancing method",
    )

    args = parser.parse_args()

    print("⚖️  SafeSpeak Dataset Balancing")
    print("=" * 40)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for dataset_name in args.datasets:
        print(f"\n🔄 Processing {dataset_name}...")

        try:
            # Load dataset
            df = load_dataset(args.data_dir, dataset_name)
            print(f"✓ Loaded {len(df)} samples")

            # Analyze current balance
            balance_before = analyze_balance(df)
            print(f"📊 Balance before: {balance_before['balance_score']:.3f}")

            # Apply balancing
            if args.method == "auto":
                balanced_df, method_used = balance_dataset_auto(df)
            elif args.method == "smote":
                balanced_df = balance_dataset_smote(df)
                method_used = "smote_oversampling"
            else:  # undersample
                balanced_df = balance_dataset_undersample(df)
                method_used = "random_undersampling"

            # Analyze balance after
            balance_after = analyze_balance(balanced_df)
            print(f"📊 Balance after: {balance_after['balance_score']:.3f}")
            print(f"📈 Samples: {len(df)} → {len(balanced_df)}")

            # Save balanced dataset
            output_file = save_balanced_dataset(
                balanced_df, args.output_dir, dataset_name, method_used
            )
            print(f"💾 Saved to: {output_file}")

            results[dataset_name] = {
                "original_samples": len(df),
                "balanced_samples": len(balanced_df),
                "method": method_used,
                "balance_before": balance_before["balance_score"],
                "balance_after": balance_after["balance_score"],
                "output_file": str(output_file),
            }

        except Exception as e:
            print(f"✗ Failed to balance {dataset_name}: {e}")
            results[dataset_name] = {"error": str(e)}

    # Save results summary
    results_file = args.output_dir / "balancing_results.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Balancing complete! Results saved to {results_file}")

    # Print summary
    print("\n📈 Summary:")
    for dataset, result in results.items():
        if "error" not in result:
            improvement = result["balance_after"] - result["balance_before"]
            print(
                f"  {dataset}: {result['balance_before']:.3f} → {result['balance_after']:.3f} "
                f"(+{improvement:.3f})"
            )
        else:
            print(f"  {dataset}: Failed - {result['error']}")


if __name__ == "__main__":
    main()
