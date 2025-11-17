#!/usr/bin/env python3
"""
Phase 4: Platinum Phase - Drift Detection & Continual Learning
Implements comprehensive drift detection for production model monitoring.
"""

import argparse
from pathlib import Path
import json
import warnings
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
import torch
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.pytorch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

warnings.filterwarnings("ignore")


class DriftDetector:
    """Comprehensive drift detection system for SafeSpeak toxicity detector."""

    def __init__(self, model_path: str, reference_data_path: str, alpha: float = 0.05):
        """
        Initialize drift detector.

        Args:
            model_path: Path to the production model
            reference_data_path: Path to reference dataset for baseline
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        self.model_path = model_path
        self.reference_data_path = reference_data_path

        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, return_dict=False
        )
        self.model.eval()

        # Load reference data for baseline
        print(f"Loading reference data from {reference_data_path}")
        self.reference_df = pd.read_csv(reference_data_path)

        # Compute reference statistics
        self._compute_reference_statistics()

        # Initialize drift history
        self.drift_history = []

    def _compute_reference_statistics(self):
        """Compute baseline statistics from reference data."""
        print("Computing reference statistics...")

        # Text statistics
        texts = self.reference_df["text"].tolist()
        self.reference_stats = {
            "text_lengths": [len(text) for text in texts],
            "vocab_size": len(set(" ".join(texts).split())),
            "avg_text_length": np.mean([len(text) for text in texts]),
            "std_text_length": np.std([len(text) for text in texts]),
        }

        # Label distribution
        self.reference_label_dist = (
            self.reference_df["label"].value_counts(normalize=True).to_dict()
        )

        # Model predictions on reference data
        print("Computing reference model predictions...")
        reference_predictions = []
        reference_confidences = []

        for text in texts[:1000]:  # Sample for efficiency
            probs = self._get_softmax_probs(text)
            pred_class = np.argmax(probs)
            confidence = np.max(probs)

            reference_predictions.append(pred_class)
            reference_confidences.append(confidence)

        self.reference_stats.update(
            {
                "reference_predictions": reference_predictions,
                "reference_confidences": reference_confidences,
                "avg_confidence": np.mean(reference_confidences),
                "confidence_std": np.std(reference_confidences),
            }
        )

        print("Reference statistics computed!")

    def _get_softmax_probs(self, text: str) -> np.ndarray:
        """Get softmax probabilities for input text."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs[0]
            probs = torch.softmax(logits, dim=1).numpy()[0]

        return probs

    def detect_data_drift(self, new_texts: List[str]) -> Dict:
        """
        Detect data drift by comparing input text statistics.

        Args:
            new_texts: New input texts to analyze

        Returns:
            Dictionary with drift detection results
        """
        print(f"Analyzing {len(new_texts)} new texts for data drift...")

        # Compute current statistics
        current_lengths = [len(text) for text in new_texts]
        current_vocab = len(set(" ".join(new_texts).split()))

        current_stats = {
            "avg_text_length": np.mean(current_lengths),
            "std_text_length": np.std(current_lengths),
            "vocab_size": current_vocab,
        }

        # Statistical tests for drift detection
        drift_results = {}

        # Text length drift (Kolmogorov-Smirnov test)
        ks_stat, ks_p_value = stats.ks_2samp(
            self.reference_stats["text_lengths"],
            current_lengths,
            alternative="two-sided",
        )

        drift_results["text_length_drift"] = {
            "test": "Kolmogorov-Smirnov",
            "statistic": ks_stat,
            "p_value": ks_p_value,
            "drift_detected": ks_p_value < self.alpha,
            "reference_avg": self.reference_stats["avg_text_length"],
            "current_avg": current_stats["avg_text_length"],
            "change_percent": (
                (
                    current_stats["avg_text_length"]
                    - self.reference_stats["avg_text_length"]
                )
                / self.reference_stats["avg_text_length"]
                * 100
            ),
        }

        # Vocabulary size change
        vocab_change = (
            (current_vocab - self.reference_stats["vocab_size"])
            / self.reference_stats["vocab_size"]
            * 100
        )

        drift_results["vocabulary_drift"] = {
            "drift_detected": abs(vocab_change) > 20,  # 20% threshold
            "reference_vocab": self.reference_stats["vocab_size"],
            "current_vocab": current_vocab,
            "change_percent": vocab_change,
        }

        return drift_results

    def detect_label_drift(self, new_labels: List[str]) -> Dict:
        """
        Detect label drift by comparing output label distributions.

        Args:
            new_labels: New labels to analyze

        Returns:
            Dictionary with label drift detection results
        """
        print(f"Analyzing {len(new_labels)} new labels for label drift...")

        # Convert string labels to distribution
        current_label_dist = (
            pd.Series(new_labels).value_counts(normalize=True).to_dict()
        )

        # Ensure all classes are represented
        for label in self.reference_label_dist.keys():
            if label not in current_label_dist:
                current_label_dist[label] = 0.0

        # Jensen-Shannon divergence for distribution comparison
        ref_probs = np.array(
            [
                self.reference_label_dist.get(label, 0)
                for label in sorted(self.reference_label_dist.keys())
            ]
        )
        curr_probs = np.array(
            [
                current_label_dist.get(label, 0)
                for label in sorted(self.reference_label_dist.keys())
            ]
        )

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_probs = np.clip(ref_probs, epsilon, 1)
        curr_probs = np.clip(curr_probs, epsilon, 1)
        ref_probs /= ref_probs.sum()
        curr_probs /= curr_probs.sum()

        jsd = jensenshannon(ref_probs, curr_probs)

        # Chi-square test for categorical distributions
        ref_counts = np.array(
            [
                self.reference_label_dist.get(label, 0) * len(new_labels)
                for label in sorted(self.reference_label_dist.keys())
            ]
        )
        curr_counts = np.array(
            [
                current_label_dist.get(label, 0) * len(new_labels)
                for label in sorted(self.reference_label_dist.keys())
            ]
        )

        try:
            chi_stat, chi_p_value = stats.chisquare(curr_counts, ref_counts)
        except ValueError:
            chi_stat, chi_p_value = float("inf"), 0.0

        drift_results = {
            "distribution_drift": {
                "test": "Jensen-Shannon Divergence",
                "statistic": jsd,
                "drift_detected": jsd > 0.1,  # JSD threshold
                "reference_dist": self.reference_label_dist,
                "current_dist": current_label_dist,
            },
            "categorical_drift": {
                "test": "Chi-Square",
                "statistic": chi_stat,
                "p_value": chi_p_value,
                "drift_detected": chi_p_value < self.alpha,
            },
        }

        return drift_results

    def detect_concept_drift(self, new_texts: List[str], new_labels: List[str]) -> Dict:
        """
        Detect concept drift by monitoring model performance degradation.

        Args:
            new_texts: New input texts
            new_labels: Corresponding true labels

        Returns:
            Dictionary with concept drift detection results
        """
        print(f"Analyzing {len(new_texts)} samples for concept drift...")

        # Get model predictions
        predictions = []
        confidences = []

        for text in new_texts:
            probs = self._get_softmax_probs(text)
            pred_class = np.argmax(probs)
            confidence = np.max(probs)

            predictions.append(pred_class)
            confidences.append(confidence)

        # Label encoding (assuming same as training)
        label_mapping = {"Neutral": 0, "Toxic": 1, "Hate": 2, "Threat": 3}
        true_labels_encoded = [label_mapping.get(label, 0) for label in new_labels]

        # Performance metrics
        from sklearn.metrics import accuracy_score, f1_score

        accuracy = accuracy_score(true_labels_encoded, predictions)
        macro_f1 = f1_score(true_labels_encoded, predictions, average="macro")

        # Confidence distribution comparison
        current_avg_confidence = np.mean(confidences)
        current_confidence_std = np.std(confidences)

        confidence_drift = abs(
            current_avg_confidence - self.reference_stats["avg_confidence"]
        )

        drift_results = {
            "performance_drift": {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "drift_detected": macro_f1 < 0.6,  # Performance threshold
                "reference_performance": "N/A",  # Would need historical data
            },
            "confidence_drift": {
                "current_avg_confidence": current_avg_confidence,
                "reference_avg_confidence": self.reference_stats["avg_confidence"],
                "confidence_change": confidence_drift,
                "drift_detected": confidence_drift
                > 0.1,  # 10% confidence change threshold
            },
        }

        return drift_results

    def run_comprehensive_drift_analysis(
        self, new_texts: List[str], new_labels: Optional[List[str]] = None
    ) -> Dict:
        """
        Run comprehensive drift analysis on new data.

        Args:
            new_texts: New input texts
            new_labels: Optional true labels for supervised drift detection

        Returns:
            Complete drift analysis report
        """
        print("🔍 Running comprehensive drift analysis...")

        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(new_texts),
            "drift_detected": False,
            "data_drift": {},
            "label_drift": {},
            "concept_drift": {},
        }

        # Data drift detection
        analysis_results["data_drift"] = self.detect_data_drift(new_texts)

        # Label drift detection (if labels provided)
        if new_labels:
            analysis_results["label_drift"] = self.detect_label_drift(new_labels)

            # Concept drift detection
            analysis_results["concept_drift"] = self.detect_concept_drift(
                new_texts, new_labels
            )

        # Overall drift assessment
        drift_flags = []

        # Check data drift
        for drift_type, results in analysis_results["data_drift"].items():
            if results.get("drift_detected", False):
                drift_flags.append(f"Data drift: {drift_type}")

        # Check label drift
        for drift_type, results in analysis_results["label_drift"].items():
            if results.get("drift_detected", False):
                drift_flags.append(f"Label drift: {drift_type}")

        # Check concept drift
        for drift_type, results in analysis_results["concept_drift"].items():
            if results.get("drift_detected", False):
                drift_flags.append(f"Concept drift: {drift_type}")

        analysis_results["drift_detected"] = len(drift_flags) > 0
        analysis_results["drift_summary"] = drift_flags

        # Store in history
        self.drift_history.append(analysis_results)

        print(
            f"✅ Drift analysis complete. Drift detected: {analysis_results['drift_detected']}"
        )
        if drift_flags:
            print(f"🚨 Detected drifts: {', '.join(drift_flags)}")

        return analysis_results


def main():
    parser = argparse.ArgumentParser(
        description="Drift detection for SafeSpeak toxicity detector"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/bert_max_french_augmentation/fold_0/checkpoint-14754",
        help="Path to production model",
    )
    parser.add_argument(
        "--reference-data",
        type=str,
        default="data/final/train_corrected_french_max_augmentation.csv",
        help="Path to reference dataset for baseline",
    )
    parser.add_argument(
        "--new-data",
        type=str,
        default="data/processed/validation.csv",
        help="Path to new data for drift analysis",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/drift_detection",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Start MLflow run
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run(run_name="drift_detection_analysis"):
        # Log parameters
        mlflow.log_param("model_path", args.model_path)
        mlflow.log_param("reference_data", args.reference_data)
        mlflow.log_param("new_data", args.new_data)
        mlflow.log_param("alpha", args.alpha)

        # Initialize drift detector
        detector = DriftDetector(args.model_path, args.reference_data, args.alpha)

        # Load new data for analysis
        print(f"Loading new data from {args.new_data}")
        new_df = pd.read_csv(args.new_data)

        # Run comprehensive drift analysis
        new_texts = new_df["text"].tolist()
        new_labels = new_df["label"].tolist() if "label" in new_df.columns else None

        results = detector.run_comprehensive_drift_analysis(new_texts, new_labels)

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        results_file = output_dir / "drift_analysis_results.json"

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        serializable_results = convert_numpy_types(results)

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        # Log metrics to MLflow
        mlflow.log_metric("drift_detected", int(results["drift_detected"]))
        mlflow.log_metric("sample_size", results["sample_size"])

        # Log drift details
        if results["data_drift"]:
            for drift_type, drift_data in results["data_drift"].items():
                if "statistic" in drift_data:
                    mlflow.log_metric(
                        f"data_drift_{drift_type}_stat", drift_data["statistic"]
                    )
                if "p_value" in drift_data:
                    mlflow.log_metric(
                        f"data_drift_{drift_type}_pval", drift_data["p_value"]
                    )

        if results["concept_drift"]:
            for drift_type, drift_data in results["concept_drift"].items():
                if "macro_f1" in drift_data:
                    mlflow.log_metric("concept_drift_f1", drift_data["macro_f1"])
                if "accuracy" in drift_data:
                    mlflow.log_metric("concept_drift_accuracy", drift_data["accuracy"])

        # Log results file
        mlflow.log_artifact(str(results_file))

        print("\n🎯 Drift Detection Analysis Complete!")
        print(f"Results saved to {results_file}")
        print(f"Drift detected: {results['drift_detected']}")
        if results["drift_summary"]:
            print(f"Drift types: {', '.join(results['drift_summary'])}")
        print("Check MLflow UI for detailed metrics")


if __name__ == "__main__":
    main()
