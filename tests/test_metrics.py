from safespeak.eval.metrics import compute_metrics


def test_compute_metrics_basic():
    y_true = ["Toxic", "Neutral", "Toxic", "Hate"]
    y_pred = ["Toxic", "Neutral", "Neutral", "Hate"]
    y_proba = [
        [0.1, 0.2, 0.7],
        [0.2, 0.6, 0.2],
        [0.2, 0.5, 0.3],
        [0.7, 0.2, 0.1],
    ]
    labels = ["Hate", "Neutral", "Toxic"]

    result = compute_metrics(y_true, y_pred, y_proba=y_proba, labels=labels)

    assert 0 <= result.macro_f1 <= 1
    assert set(result.per_label_f1.keys()) == set(labels)
    assert result.log_loss is not None
    metrics_dict = result.to_dict()
    assert "macro_precision" in metrics_dict
    assert "f1_Toxic" in metrics_dict
