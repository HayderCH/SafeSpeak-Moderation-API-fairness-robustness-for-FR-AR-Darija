"""Cross-validation framework for SafeSpeak datasets."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_dataset(data_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Load a dataset."""
    processed_dir = data_dir / "processed"
    csv_file = processed_dir / f"{dataset_name}.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {csv_file}")

    return pd.read_csv(csv_file)


def analyze_current_splits(df: pd.DataFrame, split_col: str = "split") -> dict:
    """Analyze current train/dev/test splits in a dataset."""
    if split_col not in df.columns:
        return {"has_splits": False, "message": f"No '{split_col}' column found"}

    split_counts = df[split_col].value_counts()
    total = len(df)

    return {
        "has_splits": True,
        "split_distribution": dict(split_counts),
        "split_percentages": {
            k: round(v / total * 100, 2) for k, v in split_counts.items()
        },
        "total_samples": total,
    }


def create_stratified_splits(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits."""
    # First split: separate test set
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=random_state
    )

    # Second split: separate train and validation from remaining data
    val_relative_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_relative_size,
        stratify=train_val[label_col],
        random_state=random_state,
    )

    # Add split labels
    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    return train, val, test


def create_cross_validation_splits(
    df: pd.DataFrame,
    label_col: str = "label",
    n_splits: int = 5,
    random_state: int = 42,
) -> list[dict]:
    """Create cross-validation folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df[label_col])):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        train_df["split"] = "train"
        val_df["split"] = "val"

        folds.append(
            {
                "fold": fold_idx + 1,
                "train": train_df,
                "val": val_df,
                "train_samples": len(train_df),
                "val_samples": len(val_df),
            }
        )

    return folds


def save_splits(splits: dict, output_dir: Path, dataset_name: str) -> dict[str, str]:
    """Save split datasets to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    for split_name, split_df in splits.items():
        if isinstance(split_df, pd.DataFrame) and len(split_df) > 0:
            output_file = output_dir / f"{dataset_name}_{split_name}.csv"
            split_df.to_csv(output_file, index=False)
            saved_files[split_name] = str(output_file)

    return saved_files


def process_dataset_for_cv(
    data_dir: Path,
    dataset_name: str,
    output_dir: Path,
    method: str = "stratified",
    **kwargs,
) -> dict:
    """Process a single dataset for cross-validation."""
    print(f"🔄 Processing {dataset_name}...")

    # Load dataset
    df = load_dataset(data_dir, dataset_name)
    print(f"✓ Loaded {len(df)} samples")

    # Analyze current splits
    current_splits = analyze_current_splits(df)
    print(f"📊 Current splits: {current_splits}")

    results = {
        "dataset": dataset_name,
        "original_samples": len(df),
        "current_splits": current_splits,
    }

    if method == "stratified":
        # Create stratified splits
        stratified_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["test_size", "val_size", "random_state"]
        }
        train_df, val_df, test_df = create_stratified_splits(df, **stratified_kwargs)

        splits = {"train": train_df, "val": val_df, "test": test_df}

        # Save splits
        saved_files = save_splits(splits, output_dir, dataset_name)

        results.update(
            {
                "method": "stratified_splits",
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "saved_files": saved_files,
            }
        )

        print(
            f"✅ Created stratified splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test"
        )

    elif method == "cross_validation":
        # Create cross-validation folds
        cv_kwargs = {
            k: v for k, v in kwargs.items() if k in ["n_splits", "random_state"]
        }
        folds = create_cross_validation_splits(df, **cv_kwargs)

        # Save each fold
        saved_files = {}
        for fold_info in folds:
            fold_num = fold_info["fold"]
            fold_splits = {
                f"fold_{fold_num}_train": fold_info["train"],
                f"fold_{fold_num}_val": fold_info["val"],
            }
            fold_files = save_splits(fold_splits, output_dir, dataset_name)
            saved_files.update(fold_files)

        results.update(
            {
                "method": "cross_validation",
                "n_folds": len(folds),
                "saved_files": saved_files,
            }
        )

        print(f"✅ Created {len(folds)}-fold cross-validation")

    return results


def main():
    """Main cross-validation script."""
    parser = argparse.ArgumentParser(
        description="Create cross-validation splits for SafeSpeak datasets"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Data directory path"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/splits"),
        help="Output directory for split datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "algd_toxicity_dz",
            "arabizi_offensive_lang",
            "armi_ar_train",
            "base_donnee_hate_speech_ar",
            "hatecheck_arabic",
            "hatecheck_french",
            "hatexplain_fr",
            "narabizi_treebank",
            "sample_toxicity",
            "toxic_arabic_tweets",
            "t_hsab_tunisian",
        ],
        help="Datasets to process",
    )
    parser.add_argument(
        "--method",
        choices=["stratified", "cross_validation"],
        default="stratified",
        help="Splitting method",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set size (for stratified)"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation set size (for stratified)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds (for cross_validation)",
    )

    args = parser.parse_args()

    print("🔀 SafeSpeak Cross-Validation Framework")
    print("=" * 40)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for dataset_name in args.datasets:
        try:
            result = process_dataset_for_cv(
                args.data_dir,
                dataset_name,
                args.output_dir,
                args.method,
                test_size=args.test_size,
                val_size=args.val_size,
                n_splits=args.n_splits,
            )
            results[dataset_name] = result

        except Exception as e:
            print(f"✗ Failed to process {dataset_name}: {e}")
            results[dataset_name] = {"error": str(e)}

    # Save results summary
    results_file = args.output_dir / "cv_results.json"

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    with results_file.open("w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(results), f, indent=2, ensure_ascii=False)

    print(f"\n✅ Cross-validation setup complete! Results saved to {results_file}")

    # Print summary
    print("\n📈 Summary:")
    successful = 0
    total_samples = 0

    for dataset, result in results.items():
        if "error" not in result:
            successful += 1
            total_samples += result["original_samples"]
            method = result["method"]
            if method == "stratified_splits":
                print(
                    f"  {dataset}: {result['train_samples']}/{result['val_samples']}/{result['test_samples']} "
                    f"({method})"
                )
            else:
                print(f"  {dataset}: {result['n_folds']} folds ({method})")
        else:
            print(f"  {dataset}: Failed - {result['error']}")

    print(f"\n✅ Successfully processed {successful}/{len(results)} datasets")
    print(f"📊 Total samples processed: {total_samples}")


if __name__ == "__main__":
    main()
