"""Train transformer-based models for multilingual toxicity detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from safespeak.data.loaders import build_splits, load_dataset
from safespeak.eval.metrics import compute_metrics
from safespeak.preprocessing.normalize import normalize_text


class SafeSpeakDataset(torch.utils.data.Dataset):
    """Dataset class for SafeSpeak toxicity detection."""

    def __init__(
        self,
        texts: list[str],
        labels: list[str],
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Create label mapping
        unique_labels = sorted(set(labels))
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Preprocess text
        text = normalize_text(text, transliterate=False)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,  # Will be handled by collator
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.label2id[label], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train transformer model for toxicity detection"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to canonical CSV/JSONL dataset with split column",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xlm-roberta-base",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/transformer"),
        help="Directory to store model and metrics",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def compute_metrics_func(eval_pred: Any) -> dict[str, float]:
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Convert back to string labels for our metrics function
    # This is a simplified version - in practice you'd need the label mapping
    y_pred = [str(p) for p in predictions]
    y_true = [str(label) for label in labels]

    results = compute_metrics(y_true, y_pred)
    return results.to_dict()


def train(args: argparse.Namespace) -> None:
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    dataset = load_dataset(args.data)
    splits = build_splits(dataset)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(set(splits.train["label"])),
    )

    # Create datasets
    train_dataset = SafeSpeakDataset(
        splits.train["text"].tolist(),
        splits.train["label"].tolist(),
        tokenizer,
        args.max_length,
    )
    dev_dataset = SafeSpeakDataset(
        splits.dev["text"].tolist(),
        splits.dev["label"].tolist(),
        tokenizer,
        args.max_length,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=100,
        seed=args.seed,
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_func,
    )

    # Train
    trainer.train()

    # Save model and tokenizer
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output / "model")
    tokenizer.save_pretrained(args.output / "model")

    # Save label mappings
    label_mapping = {
        "label2id": train_dataset.label2id,
        "id2label": train_dataset.id2label,
    }
    with (args.output / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    # Final evaluation
    eval_results = trainer.evaluate()
    with (args.output / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print(f"Training complete! Model saved to {args.output}")
    print(f"Final metrics: {eval_results}")


if __name__ == "__main__":
    train(parse_args())
