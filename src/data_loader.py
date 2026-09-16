"""
Data loader for the Breast Histopathology Images (IDC) dataset.

Unlike the histopathologic-cancer-detection (PatchCamelyon-style) project,
this dataset has NO single train_labels.csv. Instead, labels come from the
folder each patch sits in:

    data/<patient_id>/0/<patch>.png   -> label 0 (IDC negative)
    data/<patient_id>/1/<patch>.png   -> label 1 (IDC positive)

This module walks that structure and returns a single dataframe with
columns: id, label, filepath, patient_id.
"""

import os
import pandas as pd


def load_labels(data_dir: str) -> pd.DataFrame:
    """Walk data_dir/<patient_id>/<0 or 1>/*.png and build a labels dataframe.

    Returns
    -------
    pd.DataFrame with columns:
        id          - filename (unique-ish, kept for parity with the
                      original PatchCamelyon-style project's schema)
        label       - 0 (negative) or 1 (positive)
        filepath    - absolute path to the .png patch
        patient_id  - top-level folder name, used for patient-level splitting
    """
    rows = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"DATA_DIR not found: {data_dir}\n"
            "Download the dataset from Kaggle and unzip it into this folder. "
            "Expected layout: data/<patient_id>/<0 or 1>/*.png"
        )

    for patient_id in sorted(os.listdir(data_dir)):
        patient_path = os.path.join(data_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue
        for label in ("0", "1"):
            label_path = os.path.join(patient_path, label)
            if not os.path.isdir(label_path):
                continue
            for fname in os.listdir(label_path):
                if fname.lower().endswith(".png"):
                    rows.append(
                        {
                            "id": fname,
                            "label": int(label),
                            "filepath": os.path.join(label_path, fname),
                            "patient_id": patient_id,
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"No .png patches found under {data_dir}. "
            "Check that the dataset was unzipped correctly."
        )
    return df


if __name__ == "__main__":
    import config

    df = load_labels(config.DATA_DIR)
    print(f"Loaded {len(df):,} patches from {df['patient_id'].nunique():,} patients")
    print(df["label"].value_counts(normalize=True).rename("proportion"))
