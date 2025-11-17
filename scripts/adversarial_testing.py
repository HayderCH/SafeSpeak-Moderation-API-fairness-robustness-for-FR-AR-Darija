#!/usr/bin/env python3
"""
Phase 3: Adversarial Stress Testing & Robustness Evaluation
Tests model robustness against typos, emojis, leetspeak, and code-switching.
"""

import argparse
from pathlib import Path
import json
import random
import re
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)


class AdversarialGenerator:
    """Generate adversarial examples for robustness testing."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        # Leetspeak mappings
        self.leetspeak = {
            "a": ["4", "@"],
            "e": ["3"],
            "i": ["1", "!"],
            "o": ["0"],
            "s": ["5", "$"],
            "t": ["7"],
            "l": ["1"],
            "z": ["2"],
        }

        # Toxic emojis that might be used in harassment
        self.toxic_emojis = ["😡", "🤬", "💥", "🔥", "😈", "👿", "💀", "☠️", "⚠️", "🚫"]

        # Code-switching patterns (French-Arabic mixing)
        self.code_switch_patterns = {
            "fr_ar": {
                "the": "ال",
                "is": "هو",
                "you": "أنت",
                "fuck": "زبي",
                "shit": "خراء",
            }
        }

    def add_typos(self, text: str, error_rate: float = 0.1) -> str:
        """Add random typos by swapping adjacent characters."""
        if len(text) < 2:
            return text

        chars = list(text)
        num_errors = max(1, int(len(chars) * error_rate))

        for _ in range(num_errors):
            i = random.randint(0, len(chars) - 2)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]

        return "".join(chars)

    def add_leetspeak(self, text: str, replace_rate: float = 0.3) -> str:
        """Replace letters with leetspeak equivalents."""
        result = []
        for char in text.lower():
            if char in self.leetspeak and random.random() < replace_rate:
                result.append(random.choice(self.leetspeak[char]))
            else:
                result.append(char)
        return "".join(result)

    def add_emojis(self, text: str, emoji_rate: float = 0.2) -> str:
        """Add toxic emojis randomly throughout text."""
        words = text.split()
        result = []

        for word in words:
            result.append(word)
            if random.random() < emoji_rate:
                result.append(random.choice(self.toxic_emojis))

        return " ".join(result)

    def add_code_switching(self, text: str, switch_rate: float = 0.2) -> str:
        """Mix languages by replacing words with equivalents."""
        words = text.split()
        result = []

        for word in words:
            if (
                word.lower() in self.code_switch_patterns["fr_ar"]
                and random.random() < switch_rate
            ):
                result.append(self.code_switch_patterns["fr_ar"][word.lower()])
            else:
                result.append(word)

        return " ".join(result)

    def elongate_words(self, text: str, elongate_rate: float = 0.1) -> str:
        """Elongate vowels for emphasis (e.g., 'fuck' -> 'fuuuck')."""

        def elongate_word(word):
            vowels = "aeiou"
            result = []
            for char in word:
                result.append(char)
                if char.lower() in vowels and random.random() < elongate_rate:
                    result.append(char)  # Double the vowel
            return "".join(result)

        return " ".join(elongate_word(word) for word in text.split())

    def generate_adversarial_samples(self, texts: List[str]) -> Dict[str, List[str]]:
        """Generate various types of adversarial examples."""
        adversarial_types = {
            "original": texts,
            "typos": [self.add_typos(text) for text in texts],
            "leetspeak": [self.add_leetspeak(text) for text in texts],
            "emojis": [self.add_emojis(text) for text in texts],
            "code_switching": [self.add_code_switching(text) for text in texts],
            "elongated": [self.elongate_words(text) for text in texts],
            "combined": [
                self.add_typos(self.add_leetspeak(self.add_emojis(text)))
                for text in texts
            ],
        }

        return adversarial_types


class AdversarialTester:
    """Test model robustness on adversarial examples."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"Loading model from {model_path}")
        # Use base model tokenizer since checkpoints don't include it
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Create pipeline for easier inference
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            return_all_scores=True,
        )

        self.generator = AdversarialGenerator()

    def predict_batch(self, texts: List[str]) -> np.ndarray:
        """Get predictions for a batch of texts."""
        predictions = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Truncate long texts to avoid token limit issues
            batch = [text[:1000] for text in batch]  # Rough character limit
            batch_preds = self.classifier(batch)

            # Convert to binary predictions (assuming toxic=1, neutral=0)
            batch_binary = []
            for pred in batch_preds:
                # Sort by score and take the highest
                pred_sorted = sorted(pred, key=lambda x: x["score"], reverse=True)
                # Assume 'LABEL_1' is toxic, 'LABEL_0' is neutral
                is_toxic = 1 if pred_sorted[0]["label"] == "LABEL_1" else 0
                batch_binary.append(is_toxic)

            predictions.extend(batch_binary)

        return np.array(predictions)

    def evaluate_robustness(self, test_df: pd.DataFrame, output_dir: Path) -> Dict:
        """Evaluate model on various adversarial perturbations."""
        print("Generating adversarial test samples...")

        # Sample a subset for testing (to keep it manageable)
        sample_size = min(1000, len(test_df))
        test_sample = test_df.sample(n=sample_size, random_state=42)

        original_texts = test_sample["text"].tolist()

        # Convert string labels to integers (1 for toxic, 0 for neutral)
        def label_to_int(label):
            return 0 if label == "Neutral" else 1

        true_labels = test_sample["label"].apply(label_to_int).values

        # Generate adversarial examples
        adversarial_data = self.generator.generate_adversarial_samples(original_texts)

        results = {}

        print("Testing model robustness...")

        for adv_type, adv_texts in adversarial_data.items():
            print(f"  Testing {adv_type}...")

            # Get predictions
            pred_labels = self.predict_batch(adv_texts)

            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average="macro", zero_division=0
            )

            results[adv_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "samples": len(adv_texts),
            }

            print(".3f")

        # Calculate robustness metrics
        original_f1 = results["original"]["f1"]
        robustness_scores = {}

        for adv_type, metrics in results.items():
            if adv_type != "original":
                f1_drop = original_f1 - metrics["f1"]
                relative_drop = f1_drop / original_f1 if original_f1 > 0 else 0
                robustness_scores[adv_type] = {
                    "f1_drop": f1_drop,
                    "relative_drop": relative_drop,
                    "f1_score": metrics["f1"],
                }

        # Overall robustness score (average relative drop)
        avg_relative_drop = np.mean(
            [score["relative_drop"] for score in robustness_scores.values()]
        )

        summary = {
            "original_performance": results["original"],
            "adversarial_results": results,
            "robustness_scores": robustness_scores,
            "overall_robustness": {
                "average_relative_f1_drop": avg_relative_drop,
                "robustness_score": 1 - avg_relative_drop,  # Higher is better
                "test_samples": sample_size,
            },
        }

        # Save detailed results
        output_file = output_dir / "adversarial_test_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\nAdversarial testing complete!")
        print(f"Results saved to {output_file}")
        print(f"Original F1: {original_f1:.3f}")
        print(f"Average F1 drop: {avg_relative_drop:.1%}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Test model robustness against adversarial examples"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/bert_max_french_augmentation/fold_0",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/final/train_corrected_french_max_augmentation.csv",
        help="Path to test data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/adversarial_testing",
        help="Output directory for results",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of samples to test",
    )

    args = parser.parse_args()

    # Start MLflow run
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run(run_name="adversarial_stress_test"):
        # Log parameters
        mlflow.log_param("model_path", args.model_path)
        mlflow.log_param("test_data", args.test_data)
        mlflow.log_param("sample_size", args.sample_size)

        # Load test data
        print(f"Loading test data from {args.test_data}")
        test_df = pd.read_csv(args.test_data)

        # Filter for test samples (assuming we have a split column or use a portion)
        # For now, use a random sample
        test_sample = test_df.sample(
            n=min(args.sample_size, len(test_df)), random_state=42
        )

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tester
        tester = AdversarialTester(args.model_path)

        # Run adversarial testing
        results = tester.evaluate_robustness(test_sample, output_dir)

        # Log metrics to MLflow
        mlflow.log_metric("original_f1", results["original_performance"]["f1"])
        mlflow.log_metric(
            "average_f1_drop", results["overall_robustness"]["average_relative_f1_drop"]
        )
        mlflow.log_metric(
            "robustness_score", results["overall_robustness"]["robustness_score"]
        )

        # Log individual adversarial results
        for adv_type, metrics in results["adversarial_results"].items():
            mlflow.log_metric(f"{adv_type}_f1", metrics["f1"])

        # Log the results file
        mlflow.log_artifact(str(output_dir / "adversarial_test_results.json"))

        print("\n🎉 Adversarial testing completed!")
        print("Check MLflow UI for detailed metrics")
        print(f"Results: {output_dir}/adversarial_test_results.json")


if __name__ == "__main__":
    main()
