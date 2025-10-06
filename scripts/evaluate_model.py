"""Evaluate trained models on test data or new datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from safespeak.data.loaders import build_splits, load_dataset
from safespeak.eval.metrics import compute_metrics
from safespeak.preprocessing.normalize import normalize_text


class ModelPredictor:
    """Predictor class for trained transformer models."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.label_mapping = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.load_model()

    def load_model(self) -> None:
        """Load the model, tokenizer, and label mapping."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()

        # Load label mapping
        mapping_path = self.model_path / "label_mapping.json"
        if mapping_path.exists():
            with mapping_path.open("r", encoding="utf-8") as f:
                self.label_mapping = json.load(f)
        else:
            # Fallback: assume binary classification
            self.label_mapping = {
                "label2id": {"0": 0, "1": 1},
                "id2label": {"0": "0", "1": "1"}
            }

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> tuple[list[str], list[float]]:
        """Predict labels and probabilities for a batch of texts."""
        predictions = []
        probabilities = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Preprocess texts
            processed_texts = [normalize_text(text, transliterate=False) for text in batch_texts]

            # Tokenize
            inputs = self.tokenizer(
                processed_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)

                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                probs_np = probs.cpu().numpy()

                # Convert to string labels
                for pred_idx, prob_array in zip(preds, probs_np):
                    pred_label = self.label_mapping["id2label"][str(pred_idx)]
                    max_prob = float(np.max(prob_array))

                    predictions.append(pred_label)
                    probabilities.append(max_prob)

        return predictions, probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on test data"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to canonical CSV/JSONL dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation results (default: model_path/eval_results.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction",
    )
    return parser.parse_args()


def evaluate_model(args: argparse.Namespace) -> None:
    # Load model
    predictor = ModelPredictor(args.model_path)

    # Load data
    dataset = load_dataset(args.data)
    splits = build_splits(dataset)

    if args.split not in splits:
        raise ValueError(f"Split '{args.split}' not found in dataset. Available: {list(splits.keys())}")

    split_data = splits[args.split]
    texts = split_data["text"].tolist()
    true_labels = split_data["label"].tolist()

    print(f"Evaluating on {len(texts)} samples from {args.split} split...")

    # Make predictions
    pred_labels, pred_probs = predictor.predict_batch(texts, args.batch_size)

    # Compute metrics
    metrics = compute_metrics(true_labels, pred_labels)

    # Prepare results
    results = {
        "model_path": str(args.model_path),
        "dataset": str(args.data),
        "split": args.split,
        "num_samples": len(texts),
        "metrics": metrics.to_dict(),
        "predictions": {
            "labels": pred_labels,
            "probabilities": pred_probs,
        }
    }

    # Save results
    if args.output is None:
        args.output = args.model_path / f"eval_{args.split}_results.json"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output}")
    print("\nMetrics:")
    for key, value in metrics.to_dict().items():
        if isinstance(value, float):
            print(".4f")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    evaluate_model(parse_args())
