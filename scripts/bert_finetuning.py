#!/usr/bin/env python3
"""
Step 6: Silver Phase - Multilingual BERT Fine-tuning
Fine-tunes XLM-RoBERTa for multilingual toxicity detection.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate
import shap
import mlflow
import mlflow.pytorch
import mlflow
import mlflow.pytorch


class MultilingualToxicityDataset(Dataset):
    """Dataset class for multilingual toxicity detection."""

    def __init__(
        self, texts: List[str], labels: List[str], tokenizer, max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class SafeSpeakBERTTrainer:
    """BERT fine-tuning framework for multilingual toxicity detection."""

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_length: int = 128,
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = None

        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Load and preprocess the dataset."""
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, low_memory=False)
        print(f"Loaded {len(df)} samples")

        # Language distribution
        lang_dist = df["language"].value_counts()
        print("Language distribution:")
        for lang, count in lang_dist.items():
            print(f"  {lang}: {count} ({count/len(df)*100:.1f}%)")

        return df

    def create_multilingual_splits(
        self, df: pd.DataFrame, n_splits: int = 3
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create stratified splits maintaining language balance."""
        print(f"Creating {n_splits} multilingual stratified splits...")

        splits = []
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )

        # Group by language for stratification
        for lang in df["language"].unique():
            lang_df = df[df["language"] == lang].copy()
            if len(lang_df) < n_splits:
                print(f"Warning: {lang} has only {len(lang_df)} samples (< {n_splits})")

        # Stratified split on combined language + label
        stratify_col = df["language"] + "_" + df["label"].astype(str)
        for train_idx, test_idx in skf.split(df, stratify_col):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            splits.append((train_df, test_df))

        print(f"Created {len(splits)} splits")
        return splits

    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """Encode string labels to integers."""
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(labels)

        return self.label_encoder.transform(labels)

    def decode_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """Decode integer labels back to strings."""
        return self.label_encoder.inverse_transform(encoded_labels)

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro"
        )

        # Try AUROC (may fail for multi-class)
        try:
            auroc = roc_auc_score(
                labels, predictions, multi_class="ovr", average="macro"
            )
        except:
            auroc = None

        return {"precision": precision, "recall": recall, "f1": f1, "auroc": auroc}

    def train_fold(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fold_idx: int,
        output_dir: Path,
    ) -> Dict:
        """Train BERT model on one fold."""
        print(f"\n--- Training Fold {fold_idx + 1} ---")

        # Encode labels
        train_labels = self.encode_labels(train_df["label"].tolist())
        test_labels = self.encode_labels(test_df["label"].tolist())

        # Create datasets
        train_dataset = MultilingualToxicityDataset(
            train_df["text"].tolist(), train_labels, self.tokenizer, self.max_length
        )

        test_dataset = MultilingualToxicityDataset(
            test_df["text"].tolist(), test_labels, self.tokenizer, self.max_length
        )

        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.label_encoder.classes_)
        )
        model.to(self.device)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir / f"fold_{fold_idx}"),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Evaluate
        print("Evaluating...")
        eval_results = trainer.evaluate()

        # Get predictions for detailed analysis
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)

        # Per-language evaluation
        per_lang_results = self._evaluate_per_language(
            test_df, pred_labels, predictions.label_ids
        )

        fold_result = {
            "fold": fold_idx + 1,
            "eval_metrics": eval_results,
            "per_language": per_lang_results,
            "predictions": pred_labels.tolist(),
            "true_labels": predictions.label_ids.tolist(),
            "test_languages": test_df["language"].tolist(),
        }

        print(f"Fold {fold_idx + 1}: Macro-F1 = {eval_results.get('eval_f1', 0):.3f}")
        return fold_result

    def _evaluate_per_language(
        self, test_df: pd.DataFrame, predictions: np.ndarray, true_labels: np.ndarray
    ) -> Dict:
        """Evaluate performance per language."""
        results = {}

        for lang in test_df["language"].unique():
            lang_mask = test_df["language"] == lang
            if lang_mask.sum() == 0:
                continue

            lang_pred = predictions[lang_mask]
            lang_true = true_labels[lang_mask]

            precision, recall, f1, support = precision_recall_fscore_support(
                lang_true, lang_pred, average="macro"
            )

            results[lang] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "samples": int(lang_mask.sum()),
            }

        return results

    def run_cross_validation(
        self,
        data_path: Path,
        n_splits: int = 3,
        output_dir: Path = Path("results/bert_finetuning"),
    ) -> Dict:
        """Run complete cross-validation."""
        print("=== SafeSpeak BERT Fine-tuning Pipeline ===")
        print(f"Model: {self.model_name}")
        print(f"Splits: {n_splits}")

        # Load data
        df = self.load_data(data_path)

        # Fit label encoder on entire dataset to handle all possible labels
        all_labels = df["label"].tolist()
        self.encode_labels(all_labels)  # This fits the encoder

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create splits
        splits = self.create_multilingual_splits(df, n_splits)

        # Train each fold
        fold_results = []
        for fold_idx, (train_df, test_df) in enumerate(splits):
            fold_result = self.train_fold(train_df, test_df, fold_idx, output_dir)
            fold_results.append(fold_result)

        # Aggregate results
        final_results = self._aggregate_results(fold_results, output_dir)

        # Save results
        self._save_results(final_results, output_dir)

        print("\n=== FINAL RESULTS SUMMARY ===")
        print(f"XLM-RoBERTa Model:")
        print(
            f"Macro-F1: {final_results['mean_f1']:.3f} ± {final_results['std_f1']:.3f}"
        )

        if final_results["mean_auroc"]:
            print(
                f"AUROC: {final_results['mean_auroc']:.3f} ± {final_results['std_auroc']:.3f}"
            )

        print("Per-language F1:")
        for lang, metrics in final_results["per_language_aggregated"].items():
            print(f"  {lang}: {metrics['mean_f1']:.3f} ± {metrics['std_f1']:.3f}")

        return final_results

    def _aggregate_results(self, fold_results: List[Dict], output_dir: Path) -> Dict:
        """Aggregate results across folds."""
        # Overall metrics
        f1_scores = [r["eval_metrics"].get("eval_f1", 0) for r in fold_results]
        auroc_scores = [
            r["eval_metrics"].get("eval_auroc")
            for r in fold_results
            if r["eval_metrics"].get("eval_auroc")
        ]

        # Per-language aggregation
        languages = set()
        for result in fold_results:
            languages.update(result["per_language"].keys())

        per_lang_aggregated = {}
        for lang in languages:
            lang_f1_scores = []
            lang_samples = []

            for result in fold_results:
                if lang in result["per_language"]:
                    lang_f1_scores.append(result["per_language"][lang]["f1"])
                    lang_samples.append(result["per_language"][lang]["samples"])

            if lang_f1_scores:
                per_lang_aggregated[lang] = {
                    "mean_f1": float(np.mean(lang_f1_scores)),
                    "std_f1": float(np.std(lang_f1_scores)),
                    "mean_samples": int(np.mean(lang_samples)),
                }

        return {
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
            "mean_auroc": float(np.mean(auroc_scores)) if auroc_scores else None,
            "std_auroc": float(np.std(auroc_scores)) if auroc_scores else None,
            "per_language_aggregated": per_lang_aggregated,
            "fold_results": fold_results,
        }

    def _save_results(self, results: Dict, output_dir: Path):
        """Save results to files."""
        # Summary text
        summary_path = output_dir / "bert_finetuning_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("SafeSpeak BERT Fine-tuning Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Macro-F1: {results['mean_f1']:.3f} ± {results['std_f1']:.3f}\n")

            if results["mean_auroc"]:
                f.write(
                    f"AUROC: {results['mean_auroc']:.3f} ± {results['std_auroc']:.3f}\n"
                )

            f.write("\nPer-language Results:\n")
            for lang, metrics in results["per_language_aggregated"].items():
                f.write(
                    f"  {lang}: F1 = {metrics['mean_f1']:.3f} ± {metrics['std_f1']:.3f} "
                    f"({metrics['mean_samples']} samples)\n"
                )

        # Detailed JSON
        detailed_path = output_dir / "bert_finetuning_detailed.json"
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT for multilingual toxicity detection"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/final/train_corrected_french_max_augmentation.csv",
        help="Path to the training data",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xlm-roberta-base",
        choices=[
            "xlm-roberta-base",
            "bert-base-multilingual-cased",
            "distilbert-base-multilingual-cased",
        ],
        help="Pretrained model to fine-tune",
    )
    parser.add_argument(
        "--n-splits", type=int, default=3, help="Number of cross-validation splits"
    )
    parser.add_argument(
        "--max-length", type=int, default=128, help="Maximum sequence length"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/bert_finetuning",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("n_splits", args.n_splits)
        mlflow.log_param("max_length", args.max_length)
        mlflow.log_param("data_path", args.data_path)

        # Initialize trainer
        trainer = SafeSpeakBERTTrainer(
            model_name=args.model_name, max_length=args.max_length
        )

        # Run cross-validation
        results = trainer.run_cross_validation(
            Path(args.data_path),
            n_splits=args.n_splits,
            output_dir=Path(args.output_dir),
        )

        # Log final metrics
        mlflow.log_metric("mean_macro_f1", results["mean_f1"])
        mlflow.log_metric("std_macro_f1", results["std_f1"])
        mlflow.log_metric("mean_auroc", results["mean_auroc"])
        mlflow.log_metric("std_auroc", results["std_auroc"])

        # Log per-language metrics
        for lang, metrics in results["per_language"].items():
            mlflow.log_metric(f"{lang}_f1", metrics["mean_f1"])
            mlflow.log_metric(f"{lang}_precision", metrics["mean_precision"])
            mlflow.log_metric(f"{lang}_recall", metrics["mean_recall"])

    print("\n🎉 BERT fine-tuning completed!")
    print(f"Check results in {args.output_dir}")
    print("MLflow run logged - check 'mlruns' directory or MLflow UI")


if __name__ == "__main__":
    main()
