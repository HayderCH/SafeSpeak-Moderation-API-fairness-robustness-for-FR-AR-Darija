#!/usr/bin/env python3
"""
Phase 4: Platinum Phase - Conformal Prediction for Uncertainty Quantification
Implements conformal prediction to provide reliable uncertainty estimates and abstention capabilities.
"""

import argparse
from pathlib import Path
import json
import warnings
from typing import List, Dict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class ConformalPredictor:
    """Conformal prediction wrapper for BERT model using standard CP
    approach."""

    def __init__(self, model_path: str, alpha: float = 0.1):
        """
        Initialize conformal predictor.

        Args:
            model_path: Path to fine-tuned model
            alpha: Significance level (1 - confidence level)
        """
        self.alpha = alpha
        self.model_path = model_path

        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, return_dict=False
        )
        self.model.eval()

        self.calibration_scores = None
        self.is_calibrated = False

    def _get_softmax_probs(self, text: str) -> np.ndarray:
        """Get softmax probabilities for a text."""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(encoding["input_ids"], encoding["attention_mask"])[0]
            probs = torch.softmax(logits, dim=1).numpy()[0]

        return probs

    def _get_prediction_set(self, text: str) -> List[int]:
        """Get prediction set for a single text."""
        probs = self._get_softmax_probs(text)
        prediction_set = []

        for class_idx in range(len(probs)):
            nonconformity_score = self._compute_nonconformity_score(text, class_idx)
            if nonconformity_score <= self.conformal_threshold:
                prediction_set.append(class_idx)

        return prediction_set

    def _compute_nonconformity_score(self, text: str, class_idx: int) -> float:
        """Compute nonconformity score for a (text, class) pair."""
        probs = self._get_softmax_probs(text)

        # Find the highest probability among classes other than class_idx
        probs_copy = probs.copy()
        probs_copy[class_idx] = -1  # Exclude the candidate class
        max_other_prob = np.max(probs_copy)

        # Multi-class nonconformity score
        lambda_reg = 2.0
        nonconformity_score = 1.0 - probs[class_idx] + lambda_reg * max_other_prob

        return nonconformity_score

    def calibrate(self, calib_texts: List[str], calib_labels: List[int]):
        """Calibrate the conformal predictor using calibration data."""
        print(f"Calibrating conformal predictor with " f"{len(calib_texts)} samples...")

        # Compute nonconformity scores for calibration data
        # Standard multi-class nonconformity score:
        # s(x,y) = 1 - p_y(x) + λ * max_{y' ≠ y} p_{y'}(x)
        # λ balances between true class prob and highest incorrect class prob
        scores = []
        lambda_reg = 2.0  # Increased regularization parameter

        for text, true_label in zip(calib_texts, calib_labels):
            probs = self._get_softmax_probs(text)

            # Find the highest probability among incorrect classes
            probs_copy = probs.copy()
            probs_copy[true_label] = -1  # Exclude true class
            max_incorrect_prob = np.max(probs_copy)

            # Multi-class nonconformity score
            nonconformity_score = (
                1.0 - probs[true_label] + lambda_reg * max_incorrect_prob
            )
            scores.append(nonconformity_score)

        self.calibration_scores = np.array(scores)

        # Compute the (1-alpha) quantile of calibration scores
        self.conformal_threshold = np.quantile(self.calibration_scores, 1 - self.alpha)

        self.is_calibrated = True
        print(f"Calibration complete! Threshold: " f"{self.conformal_threshold:.3f}")
        print(
            f"Nonconformity scores range: "
            f"[{self.calibration_scores.min():.3f}, "
            f"{self.calibration_scores.max():.3f}]"
        )
        print(f"90th percentile: " f"{np.percentile(self.calibration_scores, 90):.3f}")

    def predict_with_uncertainty(self, texts: List[str]) -> List[Dict]:
        """
        Make predictions with uncertainty quantification.

        Returns:
            List of prediction results with confidence sets and
            uncertainty metrics
        """
        if not self.is_calibrated:
            raise ValueError(
                "Predictor must be calibrated before " "making predictions"
            )

        results = []

        for text in texts:
            probs = self._get_softmax_probs(text)
            confidence = np.max(probs)

            # Get prediction set based on conformal threshold
            prediction_set = self._get_prediction_set(text)

            # Ensure at least one prediction if confidence is high enough
            if not prediction_set and confidence >= (1.0 - self.conformal_threshold):
                # Fallback: include the most confident prediction
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    logits = self.model(
                        encoding["input_ids"], encoding["attention_mask"]
                    )[0]
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                prediction_set = [np.argmax(probs)]

            predicted_class = prediction_set[0] if prediction_set else 0
            set_size = len(prediction_set)
            is_singleton = set_size == 1
            uncertainty_score = 1.0 - confidence

            result = {
                "text": text,
                "prediction_set": prediction_set,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "uncertainty_score": uncertainty_score,
                "set_size": set_size,
                "is_singleton": is_singleton,
                "should_abstain": set_size > 1,
            }

            results.append(result)

        return results

    def evaluate_conformal_predictions(
        self,
        test_texts: List[str],
        test_labels: List[int],
        coverage_target: float = 0.9,
    ) -> Dict:
        """Evaluate conformal prediction performance."""
        print("Evaluating conformal predictions...")

        predictions = self.predict_with_uncertainty(test_texts)

        # Calculate coverage metrics
        total_predictions = len(predictions)
        covered_predictions = 0
        singleton_predictions = 0
        abstained_predictions = 0

        correct_predictions = 0
        correct_singleton = 0

        for pred, true_label in zip(predictions, test_labels):
            prediction_set = pred["prediction_set"]

            # Coverage: true label in prediction set
            if true_label in prediction_set:
                covered_predictions += 1

            # Singleton predictions
            if pred["is_singleton"]:
                singleton_predictions += 1
                if pred["predicted_class"] == true_label:
                    correct_singleton += 1

            # Abstention
            if pred["should_abstain"]:
                abstained_predictions += 1

            # Overall accuracy
            if pred["predicted_class"] == true_label:
                correct_predictions += 1

        # Calculate metrics
        coverage = covered_predictions / total_predictions
        singleton_rate = singleton_predictions / total_predictions
        abstention_rate = abstained_predictions / total_predictions
        accuracy = correct_predictions / total_predictions
        singleton_accuracy = (
            correct_singleton / singleton_predictions
            if singleton_predictions > 0
            else 0
        )

        results = {
            "coverage": coverage,
            "target_coverage": coverage_target,
            "coverage_satisfied": coverage >= coverage_target,
            "singleton_rate": singleton_rate,
            "abstention_rate": abstention_rate,
            "overall_accuracy": accuracy,
            "singleton_accuracy": singleton_accuracy,
            "efficiency": singleton_rate,
            "total_predictions": total_predictions,
            "alpha": self.alpha,
        }

        print("\nConformal Prediction Results:")
        print(f"Coverage: {coverage:.3f}")
        print(f"Singleton Rate: {singleton_rate:.3f}")
        print(f"Abstention Rate: {abstention_rate:.3f}")
        print(f"Overall Accuracy: {accuracy:.3f}")
        print(f"Singleton Accuracy: {singleton_accuracy:.3f}")

        return results


class UncertaintyQuantifier:
    """High-level interface for uncertainty quantification."""

    def __init__(self, model_path: str, alpha: float = 0.1):
        self.model_path = Path(model_path)
        self.alpha = alpha
        self.predictor = None

    def setup_conformal_prediction(
        self, calib_texts: List[str], calib_labels: List[int]
    ):
        """Set up conformal prediction with calibration."""
        self.predictor = ConformalPredictor(str(self.model_path), alpha=self.alpha)
        self.predictor.calibrate(calib_texts, calib_labels)

    def predict_with_abstention(
        self, texts: List[str], abstention_threshold: float = 0.8
    ) -> List[Dict]:
        """
        Make predictions with optional abstention based on confidence.

        Args:
            texts: Input texts to classify
            abstention_threshold: Minimum confidence required (0-1)

        Returns:
            Predictions with abstention decisions
        """
        if self.predictor is None:
            raise ValueError("Conformal predictor not initialized")

        predictions = self.predictor.predict_with_uncertainty(texts)

        # Apply abstention logic
        for pred in predictions:
            confidence = pred["confidence"]
            set_size = pred["set_size"]

            # Abstain if confidence too low OR prediction set too large
            should_abstain = (
                confidence < abstention_threshold
                or set_size > 1  # Conformal prediction abstention
            )

            pred["abstention_decision"] = should_abstain
            pred["abstention_reason"] = []
            if confidence < abstention_threshold:
                pred["abstention_reason"].append(f"low_confidence_{confidence:.3f}")
            if set_size > 1:
                pred["abstention_reason"].append(f"uncertain_set_size_{set_size}")

        return predictions

    def evaluate_uncertainty_system(
        self,
        test_texts: List[str],
        test_labels: List[int],
        abstention_threshold: float = 0.8,
    ) -> Dict:
        """Evaluate the complete uncertainty quantification system."""
        predictions = self.predict_with_abstention(test_texts, abstention_threshold)

        # Calculate metrics
        total = len(predictions)
        abstained = sum(1 for p in predictions if p["abstention_decision"])
        answered = total - abstained

        # Accuracy on answered questions
        correct_answered = 0
        for pred, true_label in zip(predictions, test_labels):
            if not pred["abstention_decision"]:
                if pred["predicted_class"] == true_label:
                    correct_answered += 1

        accuracy_answered = correct_answered / answered if answered > 0 else 0
        coverage = answered / total

        results = {
            "total_predictions": total,
            "abstained_predictions": abstained,
            "answered_predictions": answered,
            "abstention_rate": abstained / total,
            "coverage": coverage,
            "accuracy_on_answered": accuracy_answered,
            "abstention_threshold": abstention_threshold,
            "alpha": self.alpha,
        }

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Conformal prediction for uncertainty quantification"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/bert_max_french_augmentation/fold_0/checkpoint-14754",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/final/train_corrected_french_max_augmentation.csv",
        help="Path to dataset for calibration and testing",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Significance level for conformal prediction (1 - confidence)",
    )
    parser.add_argument(
        "--abstention-threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for abstention",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/conformal_prediction",
        help="Output directory for results",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=500,
        help="Size of calibration set",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Size of test set",
    )

    args = parser.parse_args()

    # Start MLflow run
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run(run_name="conformal_prediction_uncertainty"):
        # Log parameters
        mlflow.log_param("model_path", args.model_path)
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("abstention_threshold", args.abstention_threshold)
        mlflow.log_param("calib_size", args.calib_size)
        mlflow.log_param("test_size", args.test_size)

        # Load and prepare data
        print(f"Loading data from {args.data_path}")
        df = pd.read_csv(args.data_path)

        # Filter out rare classes for conformal prediction
        class_counts = df["label"].value_counts()
        valid_classes = class_counts[class_counts >= 10].index
        df = df[df["label"].isin(valid_classes)].copy()

        # Convert labels to integers (multi-class)
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        df["label_int"] = label_encoder.fit_transform(df["label"])

        # Split data (random split to avoid stratified issues)
        train_df, temp_df = train_test_split(
            df, test_size=args.calib_size + args.test_size, random_state=42
        )
        calib_df, test_df = train_test_split(
            temp_df, test_size=args.test_size, random_state=42
        )

        print(
            f"Train: {len(train_df)}, Calib: {len(calib_df)}, " f"Test: {len(test_df)}"
        )

        # Initialize uncertainty quantifier
        quantifier = UncertaintyQuantifier(args.model_path, alpha=args.alpha)

        # Setup conformal prediction
        quantifier.setup_conformal_prediction(
            calib_df["text"].tolist(), calib_df["label_int"].tolist()
        )

        # Evaluate conformal predictions
        conformal_results = quantifier.predictor.evaluate_conformal_predictions(
            test_df["text"].tolist(), test_df["label_int"].tolist()
        )

        # Evaluate uncertainty system with abstention
        uncertainty_results = quantifier.evaluate_uncertainty_system(
            test_df["text"].tolist(),
            test_df["label_int"].tolist(),
            abstention_threshold=args.abstention_threshold,
        )

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        results = {
            "conformal_prediction_results": conformal_results,
            "uncertainty_system_results": uncertainty_results,
            "configuration": {
                "alpha": args.alpha,
                "abstention_threshold": args.abstention_threshold,
                "calib_size": args.calib_size,
                "test_size": args.test_size,
            },
        }

        results_file = output_dir / "conformal_prediction_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Log metrics to MLflow
        mlflow.log_metric("conformal_coverage", conformal_results["coverage"])
        mlflow.log_metric("singleton_rate", conformal_results["singleton_rate"])
        mlflow.log_metric("abstention_rate", uncertainty_results["abstention_rate"])
        mlflow.log_metric(
            "accuracy_answered", uncertainty_results["accuracy_on_answered"]
        )
        mlflow.log_metric("system_coverage", uncertainty_results["coverage"])

        # Log results file
        mlflow.log_artifact(str(results_file))

        print("\n🎉 Conformal prediction evaluation complete!")
        print(f"Results saved to {results_file}")
        print("Check MLflow UI for detailed metrics")
        print("\nKey Results:")
        print(f"Coverage: {conformal_results['coverage']:.3f}")
        print(f"Abstention Rate: {uncertainty_results['abstention_rate']:.3f}")
        print(
            f"Accuracy on Answered: "
            f"{uncertainty_results['accuracy_on_answered']:.3f}"
        )
        print(f"System Coverage: {uncertainty_results['coverage']:.3f}")


if __name__ == "__main__":
    main()
