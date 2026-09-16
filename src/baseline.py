"""
Color histogram + Logistic Regression baseline.

Gives a simple, fast reference point to compare the CNN against. Extracts
a color-histogram feature vector from each patch and fits a logistic
regression classifier on top.
"""

import numpy as np
import joblib
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


def _extract_color_histogram(filepath: str, bins: int = 16) -> np.ndarray:
    """Extract a per-channel color histogram from the full patch.

    (IDC patches are 50x50 with no separate "center region" labeling rule
    the way PatchCamelyon patches do, so the baseline uses the whole patch.)
    """
    img = Image.open(filepath).convert("RGB")
    arr = np.array(img)
    hist = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 255), density=True)
        hist.append(h)
    return np.concatenate(hist)


def build_features(df, bins: int = 16) -> np.ndarray:
    return np.stack([_extract_color_histogram(fp, bins) for fp in df["filepath"]])


def train_baseline(train_df, val_df, model_path: str, bins: int = 16):
    X_train = build_features(train_df, bins)
    y_train = train_df["label"].values

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    X_val = build_features(val_df, bins)
    y_val = val_df["label"].values
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    val_acc = accuracy_score(y_val, val_probs > 0.5)

    print(f"[baseline] Val ROC AUC: {val_auc:.4f}  Val Accuracy: {val_acc:.4f}")

    joblib.dump({"model": clf, "bins": bins}, model_path)
    print(f"[baseline] Saved to {model_path}")
    return clf


def evaluate_baseline(test_df, model_path: str):
    bundle = joblib.load(model_path)
    clf, bins = bundle["model"], bundle["bins"]

    X_test = build_features(test_df, bins)
    y_test = test_df["label"].values
    test_probs = clf.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, test_probs)
    test_acc = accuracy_score(y_test, test_probs > 0.5)
    print(f"[baseline] Test ROC AUC: {test_auc:.4f}  Test Accuracy: {test_acc:.4f}")
    return {"roc_auc": test_auc, "accuracy": test_acc}
