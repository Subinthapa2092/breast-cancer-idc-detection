"""
Evaluate a trained CNN (and optionally the baseline) on the untouched
patient-level test split. Produces ROC AUC, accuracy, a classification
report, a ROC curve plot, and a confusion matrix plot in outputs/.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import config
from src.data_loader import load_labels
from src.preprocessing import make_patient_level_splits
from src.dataset import make_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=config.CNN_MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_labels(config.DATA_DIR)
    _, _, test_df = make_patient_level_splits(
        df, config.VAL_SIZE, config.TEST_SIZE, config.RANDOM_STATE
    )
    print(f"Evaluating on {len(test_df):,} test patches (untouched, patient-level split)")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"No model found at {args.model_path}. Train one first with src.train."
        )

    model = tf.keras.models.load_model(args.model_path)
    test_ds = make_dataset(test_df, config.IMAGE_SIZE, config.BATCH_SIZE, training=False)

    y_true = test_df["label"].values
    y_prob = model.predict(test_ds).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    test_auc = roc_auc_score(y_true, y_prob)
    test_acc = accuracy_score(y_true, y_pred)
    print(f"\nTest ROC AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\n" + classification_report(y_true, y_pred, target_names=["Negative", "IDC"]))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"CNN (AUC = {test_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - IDC Breast Cancer CNN")
    plt.legend()
    roc_path = os.path.join(config.OUTPUTS_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "IDC"])
    disp.plot(cmap="Blues")
    cm_path = os.path.join(config.OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {roc_path}")
    print(f"Saved: {cm_path}")


if __name__ == "__main__":
    main()
