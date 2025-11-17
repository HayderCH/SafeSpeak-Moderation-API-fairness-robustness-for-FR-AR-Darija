#!/usr/bin/env python3
"""
Step 5: Cross-Validation Framework for SafeSpeak
Implements Bronze phase baseline models with proper multilingual evaluation.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


class SafeSpeakCrossValidator:
    """Cross-validation framework for multilingual toxicity detection."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.vectorizer = None
        self.label_encoder = None
        self.models = {}

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Load and preprocess the dataset."""
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        # Filter out English (too few samples)
        df = df[df["language"] != "en"].copy()

        print(f"Loaded {len(df)} samples")
        print("Language distribution:")
        for lang, count in df["language"].value_counts().items():
            pct = (count / len(df)) * 100
            print(f"  {lang}: {count} ({pct:.1f}%)")

        return df

    def create_multilingual_splits(
        self, df: pd.DataFrame, n_splits: int = 5
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create stratified splits that maintain language balance."""
        splits = []

        # Create stratified splits for each language separately
        languages = df["language"].unique()

        for lang in languages:
            lang_df = df[df["language"] == lang].copy()
            lang_labels = lang_df["label"].values

            # Use stratified k-fold for this language
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )

            lang_splits = []
            for train_idx, test_idx in skf.split(lang_df, lang_labels):
                train_fold = lang_df.iloc[train_idx]
                test_fold = lang_df.iloc[test_idx]
                lang_splits.append((train_fold, test_fold))

            splits.append(lang_splits)

        # Combine splits across languages for each fold
        combined_splits = []
        for fold_idx in range(n_splits):
            train_frames = []
            test_frames = []

            for lang_splits in splits:
                train_fold, test_fold = lang_splits[fold_idx]
                train_frames.append(train_fold)
                test_frames.append(test_fold)

            train_combined = pd.concat(train_frames, ignore_index=True)
            test_combined = pd.concat(test_frames, ignore_index=True)

            combined_splits.append((train_combined, test_combined))

        return combined_splits

    def preprocess_text(self, texts: List[str]) -> np.ndarray:
        """Preprocess text data for model input."""
        # Simple preprocessing for baseline
        processed = []
        for text in texts:
            if isinstance(text, str):
                # Basic cleaning
                text = text.strip()
                # Keep emojis and special chars for now
                processed.append(text)
            else:
                processed.append("")

        return np.array(processed)

    def fit_vectorizer(self, texts: List[str]) -> None:
        """Fit TF-IDF vectorizer on training data."""
        print("Fitting TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",  # Handle Arabic/French accents
            lowercase=True,
        )

        self.vectorizer.fit(texts)
        print(
            f"Vectorizer fitted with {len(self.vectorizer.get_feature_names_out())} features"
        )

    def transform_texts(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer."""
        processed = self.preprocess_text(texts)
        return self.vectorizer.transform(processed)

    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """Encode string labels to integers."""
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(labels)

        return self.label_encoder.transform(labels)

    def decode_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """Decode integer labels back to strings."""
        return self.label_encoder.inverse_transform(encoded_labels)

    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray, model_type: str = "lr"
    ) -> None:
        """Train a classification model."""
        print(f"Training {model_type} model...")

        if model_type == "lr":
            model = LogisticRegression(
                random_state=self.random_state, max_iter=1000, class_weight="balanced"
            )
        elif model_type == "svm":
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            model = Pipeline(
                [
                    ("scaler", StandardScaler(with_mean=False)),  # TF-IDF sparse
                    (
                        "svm",
                        SGDClassifier(
                            random_state=self.random_state,
                            class_weight="balanced",
                            loss="hinge",  # SVM loss
                            penalty="l2",
                            alpha=0.0001,  # Like C=1.0 in SVC
                            max_iter=1000,
                            tol=1e-3,
                        ),
                    ),
                ]
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        self.models[model_type] = model
        print(f"{model_type.upper()} model trained")

    def evaluate_model(
        self,
        model_type: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_df: pd.DataFrame,
    ) -> Dict:
        """Evaluate model performance."""
        model = self.models[model_type]
        y_pred = model.predict(X_test)

        # Get prediction probabilities for AUROC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = model.decision_function(X_test)

        # Overall metrics
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        labels = self.label_encoder.classes_

        # Calculate AUROC (handle multi-class)
        try:
            if len(labels) == 2:
                auroc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                auroc = roc_auc_score(
                    y_test, y_prob, multi_class="ovr", average="macro"
                )
        except (ValueError, TypeError):
            auroc = None

        # Per-language metrics
        lang_metrics = {}
        for lang in test_df["language"].unique():
            lang_mask = test_df["language"] == lang
            if lang_mask.sum() > 0:
                lang_y_true = y_test[lang_mask]
                lang_y_pred = y_pred[lang_mask]

                precision, recall, f1, _ = precision_recall_fscore_support(
                    lang_y_true, lang_y_pred, average="macro", zero_division=0
                )
                lang_metrics[lang] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "samples": len(lang_y_true),
                }

        results = {
            "macro_f1": macro_f1,
            "auroc": auroc,
            "per_language": lang_metrics,
            "classification_report": classification_report(
                y_test,
                y_pred,
                labels=np.unique(np.concatenate([y_test, y_pred])),
                target_names=self.label_encoder.inverse_transform(
                    np.unique(np.concatenate([y_test, y_pred]))
                ),
                zero_division=0,
            ),
        }

        return results

    def run_cross_validation(
        self, data_path: Path, model_types: List[str] = ["lr", "svm"], n_splits: int = 5
    ) -> Dict:
        """Run complete cross-validation pipeline."""
        print("=== SafeSpeak Cross-Validation Pipeline ===")
        print(f"Model types: {model_types}")
        print(f"Number of splits: {n_splits}")
        print()

        # Load data
        df = self.load_data(data_path)

        # Create multilingual stratified splits
        print("Creating multilingual stratified splits...")
        splits = self.create_multilingual_splits(df, n_splits)

        # Encode labels
        all_labels = df["label"].values
        encoded_labels = self.encode_labels(all_labels)

        # Fit vectorizer on entire dataset
        all_texts = self.preprocess_text(df["text"].values)
        self.fit_vectorizer(all_texts)

        results = {}

        for model_type in model_types:
            print(f"\n--- Cross-Validation for {model_type.upper()} ---")
            fold_results = []

            for fold_idx, (train_df, test_df) in enumerate(splits):
                print(f"Fold {fold_idx + 1}/{n_splits}")

                # Prepare data for this fold
                X_train = self.transform_texts(train_df["text"].values)
                X_test = self.transform_texts(test_df["text"].values)

                y_train = self.encode_labels(train_df["label"].values)
                y_test = self.encode_labels(test_df["label"].values)

                # Train model
                self.train_model(X_train, y_train, model_type)

                # Evaluate
                fold_result = self.evaluate_model(model_type, X_test, y_test, test_df)
                fold_results.append(fold_result)

                print(
                    f"  Fold {fold_idx + 1}: Macro-F1 = {fold_result['macro_f1']:.3f}"
                )
            # Aggregate results
            macro_f1_scores = [r["macro_f1"] for r in fold_results]
            auroc_scores = [r["auroc"] for r in fold_results if r["auroc"] is not None]

            results[model_type] = {
                "fold_results": fold_results,
                "mean_macro_f1": np.mean(macro_f1_scores),
                "std_macro_f1": np.std(macro_f1_scores),
                "mean_auroc": np.mean(auroc_scores) if auroc_scores else None,
                "std_auroc": np.std(auroc_scores) if auroc_scores else None,
                "per_language_aggregated": self._aggregate_lang_metrics(fold_results),
            }

        return results

    def _aggregate_lang_metrics(self, fold_results: List[Dict]) -> Dict:
        """Aggregate per-language metrics across folds."""
        lang_aggregated = {}

        # Collect all languages
        all_langs = set()
        for fold_result in fold_results:
            all_langs.update(fold_result["per_language"].keys())

        for lang in all_langs:
            lang_f1_scores = []
            lang_samples = []

            for fold_result in fold_results:
                if lang in fold_result["per_language"]:
                    lang_f1_scores.append(fold_result["per_language"][lang]["f1"])
                    lang_samples.append(fold_result["per_language"][lang]["samples"])

            if lang_f1_scores:
                lang_aggregated[lang] = {
                    "mean_f1": np.mean(lang_f1_scores),
                    "std_f1": np.std(lang_f1_scores),
                    "total_samples": sum(lang_samples),
                }

        return lang_aggregated

    def save_results(self, results: Dict, output_dir: Path) -> None:
        """Save cross-validation results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_dir / "cross_validation_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=== SafeSpeak Cross-Validation Results ===\n\n")

            for model_type, model_results in results.items():
                f.write(f"Model: {model_type.upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(".3f")
                f.write(".3f")
                if model_results["mean_auroc"] is not None:
                    f.write(".3f")
                    f.write(".3f")
                f.write("\n")

                f.write("Per-Language Performance:\n")
                for lang, metrics in model_results["per_language_aggregated"].items():
                    f.write(".3f")
                f.write("\n\n")

        # Save detailed results
        import json

        detailed_path = output_dir / "cross_validation_detailed.json"
        with open(detailed_path, "w", encoding="utf-8") as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for model_type, model_results in results.items():
                json_results[model_type] = {
                    "mean_macro_f1": float(model_results["mean_macro_f1"]),
                    "std_macro_f1": float(model_results["std_macro_f1"]),
                    "mean_auroc": (
                        float(model_results["mean_auroc"])
                        if model_results["mean_auroc"]
                        else None
                    ),
                    "std_auroc": (
                        float(model_results["std_auroc"])
                        if model_results["std_auroc"]
                        else None
                    ),
                    "per_language_aggregated": {
                        lang: {
                            "mean_f1": float(metrics["mean_f1"]),
                            "std_f1": float(metrics["std_f1"]),
                            "total_samples": int(metrics["total_samples"]),
                        }
                        for lang, metrics in model_results[
                            "per_language_aggregated"
                        ].items()
                    },
                }
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-validation for SafeSpeak baseline models"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/final/train_corrected.csv"),
        help="Path to training data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cross_validation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lr", "svm"],
        choices=["lr", "svm"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, help="Number of cross-validation splits"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )

    args = parser.parse_args()

    # Run cross-validation
    cv = SafeSpeakCrossValidator(random_state=args.random_state)
    results = cv.run_cross_validation(args.data, args.models, args.n_splits)

    # Save results
    cv.save_results(results, args.output_dir)

    # Print summary
    print("\n=== FINAL RESULTS SUMMARY ===")
    for model_type, model_results in results.items():
        print(f"\n{model_type.upper()} Model:")
        print(".3f")
        if model_results["mean_auroc"]:
            print(".3f")
        print("Per-language F1:")
        for lang, metrics in model_results["per_language_aggregated"].items():
            print(".3f")


if __name__ == "__main__":
    main()
