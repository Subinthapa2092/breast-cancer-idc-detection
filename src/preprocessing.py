"""
Splitting and class-weighting utilities.

IMPORTANT: this dataset has multiple patches per patient. A naive row-level
train/val/test split leaks patches from the same patient across splits,
which inflates reported metrics (the model can partially memorize a
patient's tissue/staining style rather than generalizing). All splitting
here is done at the PATIENT level, so every patch from a given patient
ends up in exactly one of train/val/test.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_patient_level_splits(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    random_state: int,
):
    """Split a labels dataframe into train/val/test by patient_id.

    Parameters
    ----------
    df : dataframe with at least a 'patient_id' column
    val_size, test_size : fractions of PATIENTS (not rows) held out
    random_state : for reproducibility

    Returns
    -------
    train_df, val_df, test_df
    """
    patient_ids = df["patient_id"].unique()

    train_ids, temp_ids = train_test_split(
        patient_ids, test_size=(val_size + test_size), random_state=random_state
    )
    relative_test = test_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=relative_test, random_state=random_state
    )

    train_df = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)

    # Sanity check: no patient should appear in more than one split
    assert set(train_df["patient_id"]).isdisjoint(set(val_df["patient_id"]))
    assert set(train_df["patient_id"]).isdisjoint(set(test_df["patient_id"]))
    assert set(val_df["patient_id"]).isdisjoint(set(test_df["patient_id"]))

    return train_df, val_df, test_df


def compute_class_weights(train_df: pd.DataFrame) -> dict:
    """Simple inverse-frequency class weighting, keyed 0/1 for Keras."""
    counts = train_df["label"].value_counts()
    total = counts.sum()
    n_classes = len(counts)
    weights = {
        int(cls): float(total / (n_classes * count)) for cls, count in counts.items()
    }
    # Ensure both keys exist even if a class is (improbably) absent
    weights.setdefault(0, 1.0)
    weights.setdefault(1, 1.0)
    return weights
