"""Train a TF-IDF + Logistic Regression baseline on the SafeSpeak dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from safespeak.preprocessing.normalize import normalize_text
from safespeak.data.loaders import load_dataset, build_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF baseline")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to canonical CSV/JSONL dataset with split column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tfidf"),
        help="Directory to store metrics and artifacts",
    )
    parser.add_argument(
        "--max-ngrams",
        type=int,
        default=2,
        help="Maximum n-gram length for TF-IDF",
    )
    parser.add_argument(
        "--min-df",
        type=float,
        default=1.0,
        help="Minimum document frequency for TF-IDF features",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression",
    )
    return parser.parse_args()


def resolve_min_df(min_df: float) -> int | float:
    """Interpret ``min_df`` as document count when >= 1."""

    if min_df >= 1.0:
        return int(min_df)
    return min_df


def build_pipeline(max_ngrams: int, min_df: float) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=lambda text: normalize_text(text, transliterate=False),
                    tokenizer=None,
                    lowercase=False,
                    ngram_range=(1, max_ngrams),
                    min_df=resolve_min_df(min_df),
                    max_features=5000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=200,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.data)
    splits = build_splits(dataset)

    y_train = splits.train["label"].to_numpy()
    y_dev = splits.dev["label"].to_numpy()

    pipeline = build_pipeline(args.max_ngrams, args.min_df)
    pipeline.set_params(clf__C=args.c)
    pipeline.fit(splits.train["text"].tolist(), y_train)

    dev_pred = pipeline.predict(splits.dev["text"].tolist())
    dev_proba = pipeline.predict_proba(splits.dev["text"].tolist())

    metrics_output = {
        "macro_f1": metrics.f1_score(y_dev, dev_pred, average="macro", zero_division=0),
        "macro_precision": metrics.precision_score(
            y_dev, dev_pred, average="macro", zero_division=0
        ),
        "macro_recall": metrics.recall_score(
            y_dev, dev_pred, average="macro", zero_division=0
        ),
        "log_loss": metrics.log_loss(y_dev, dev_proba, labels=pipeline.classes_),
        "classes": pipeline.classes_.tolist(),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_output, fp, indent=2)

    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    train(parse_args())
