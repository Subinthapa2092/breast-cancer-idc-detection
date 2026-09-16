"""
Central configuration for the Breast Cancer (IDC) CNN project.

Dataset: Breast Histopathology Images (IDC)
https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

Folder layout expected under DATA_DIR:
    data/<patient_id>/0/*.png   -> IDC negative patches
    data/<patient_id>/1/*.png   -> IDC positive patches
"""

import os

# --- Paths -----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CNN_MODEL_PATH = os.path.join(MODEL_DIR, "breast_idc_cnn.keras")
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "color_hist_logreg_baseline.joblib")

# --- Image / data ------------------------------------------------------------
IMAGE_SIZE = 50          # native IDC patch size (50x50 RGB)
CHANNELS = 3
NUM_CLASSES = 1           # binary, sigmoid output

# --- Splits -------------------------------------------------------------------
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

# --- Training ------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3

EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# --- Class labels ---------------------------------------------------------------
LABEL_NAMES = {0: "IDC Negative", 1: "IDC Positive"}
