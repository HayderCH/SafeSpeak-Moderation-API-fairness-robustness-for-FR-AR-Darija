"""Evaluation helpers for SafeSpeak models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn import metrics


@dataclass
class EvaluationResults:
    macro_f1: float
    macro_precision: float
    macro_recall: float
    log_loss: float | None
    per_label_f1: Dict[str, float]

    def to_dict(self) -> Dict[str, float]:
        data = {
            "macro_f1": self.macro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
        }
        if self.log_loss is not None:
            data["log_loss"] = self.log_loss
        for label, value in self.per_label_f1.items():
            data[f"f1_{label}"] = value
        return data


def compute_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    y_proba: Iterable[Iterable[float]] | None = None,
    labels: List[str] | None = None,
) -> EvaluationResults:
    y_true_arr = np.array(list(y_true))
    y_pred_arr = np.array(list(y_pred))

    macro_f1 = metrics.f1_score(y_true_arr, y_pred_arr, average="macro")
    macro_precision = metrics.precision_score(y_true_arr, y_pred_arr, average="macro")
    macro_recall = metrics.recall_score(y_true_arr, y_pred_arr, average="macro")

    unique_labels = labels or sorted(np.unique(y_true_arr))
    per_label_f1 = {
        label: metrics.f1_score(
            (y_true_arr == label).astype(int),
            (y_pred_arr == label).astype(int),
        )
        for label in unique_labels
    }

    log_loss = None
    if y_proba is not None:
        log_loss = metrics.log_loss(
            y_true_arr,
            np.array(list(y_proba)),
            labels=unique_labels,
        )

    return EvaluationResults(
        macro_f1=macro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        log_loss=log_loss,
        per_label_f1=per_label_f1,
    )
