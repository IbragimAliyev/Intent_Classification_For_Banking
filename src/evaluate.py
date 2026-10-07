import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


def evaluate(y_true, y_pred, model_name, dataset_name, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    labels = sorted(set(y_true) | set(y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")


    print(f"=== {model_name} on {dataset_name} ===")
    print(f"Accuracy: {accuracy:.4f}   Macro-F1: {macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion matrix: {model_name} on {dataset_name}")
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"confusion_{model_name}_{dataset_name}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
    }
    csv_path = os.path.join(out_dir, "metrics.csv")
    pd.DataFrame([metrics]).to_csv(
        csv_path, mode="a", header=not os.path.exists(csv_path), index=False
    )

    return metrics
