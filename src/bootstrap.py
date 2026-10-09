from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def _validate_labels(y_true, *predictions):
    """Convert label inputs to arrays and validate their shape and length."""
    y_true = np.asarray(y_true)
    predictions = tuple(np.asarray(prediction) for prediction in predictions)

    if y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("y_true must be a non-empty one-dimensional array.")
    if any(prediction.ndim != 1 or prediction.size != y_true.size for prediction in predictions):
        raise ValueError("Each prediction array must be one-dimensional and match y_true in length.")

    return y_true, predictions


def _validate_n_boot(n_boot):
    if not isinstance(n_boot, (int, np.integer)) or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer.")


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42):
    """Return the 95% bootstrap confidence interval for macro-F1."""
    y_true, (y_pred,) = _validate_labels(y_true, y_pred)
    _validate_n_boot(n_boot)

    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot, dtype=float)

    for iteration in range(n_boot):
        indices = rng.integers(0, y_true.size, size=y_true.size)
        scores[iteration] = f1_score(
            y_true[indices],
            y_pred[indices],
            average="macro",
            zero_division=0,
        )

    ci_lower, ci_upper = np.percentile(scores, [2.5, 97.5])
    return float(ci_lower), float(ci_upper)


def paired_bootstrap_diff(y_true, pred_a, pred_b, n_boot=1000, seed=42):
    """Return the 95% bootstrap interval for paired macro-F1(A) - macro-F1(B)."""
    y_true, (pred_a, pred_b) = _validate_labels(y_true, pred_a, pred_b)
    _validate_n_boot(n_boot)

    rng = np.random.default_rng(seed)
    differences = np.empty(n_boot, dtype=float)

    for iteration in range(n_boot):
        indices = rng.integers(0, y_true.size, size=y_true.size)
        score_a = f1_score(
            y_true[indices],
            pred_a[indices],
            average="macro",
            zero_division=0,
        )
        score_b = f1_score(
            y_true[indices],
            pred_b[indices],
            average="macro",
            zero_division=0,
        )
        differences[iteration] = score_a - score_b

    ci_lower, ci_upper = np.percentile(differences, [2.5, 97.5])
    return float(ci_lower), float(ci_upper)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    tfidf_path = project_root / "results" / "predictions" / "tfidf_lr_minds14.csv"
    sbert_path = project_root / "results" / "predictions" / "sbert_svm_minds14.csv"

    tfidf_df = pd.read_csv(tfidf_path)
    sbert_df = pd.read_csv(sbert_path)

    if not tfidf_df["text"].equals(sbert_df["text"]):
        raise ValueError("Prediction files do not contain the same messages in the same order.")
    if not tfidf_df["intent"].equals(sbert_df["intent"]):
        raise ValueError("Prediction files do not contain the same true labels in the same order.")

    y_true = tfidf_df["intent"].to_numpy()
    tfidf_pred = tfidf_df["pred"].to_numpy()
    sbert_pred = sbert_df["pred"].to_numpy()

    print("TF-IDF macro-F1 95% CI:", bootstrap_ci(y_true, tfidf_pred, n_boot=1000, seed=42))
    print("SBERT-SVM macro-F1 95% CI:", bootstrap_ci(y_true, sbert_pred, n_boot=1000, seed=42))
    print(
        "Paired macro-F1 difference (tfidf_lr - sbert_svm) 95% CI:",
        paired_bootstrap_diff(y_true, tfidf_pred, sbert_pred, n_boot=1000, seed=42),
    )