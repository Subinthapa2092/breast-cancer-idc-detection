# Breast Cancer (IDC) Histopathology Detection CNN

A TensorFlow and Keras deep learning project for classifying **50×50 RGB breast histopathology patches** as **IDC positive** or **IDC negative**.

This project includes a **CNN built from scratch**, a **Color Histogram and Logistic Regression baseline**, and **transfer learning** with MobileNetV2, EfficientNetB0, and ResNet50.

> **Disclaimer:** This project is for learning and research purposes only. It is not a clinical diagnostic tool.

## Live Demo

Try the deployed web app here: [breast-cancer-idc-detection-qujh.onrender.com](https://breast-cancer-idc-detection-qujh.onrender.com)

Upload a 50×50 breast histopathology patch image and the CNN will return an IDC positive or IDC negative prediction with a confidence score. The app is hosted on Render's free tier, so the first request after a period of inactivity may take up to a minute to wake the server.

## Results

The main CNN was trained on **194,268 patches**, validated on **42,589 patches**, and evaluated on an untouched **40,667 patch patient level test set**.

| Metric | CNN |
|---|---:|
| Test ROC AUC | **0.9185** |
| Test Accuracy | **0.8532** |

### Class Performance

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.90 | 0.87 | 0.89 | 26,662 |
| IDC | 0.77 | **0.82** | 0.79 | 14,005 |

Class weighting was used during training to prioritize **IDC recall**.

A smaller configuration using **20,000 training patches and 3,000 validation patches** is also available for faster CPU experimentation.

## Dataset

This project uses the **Breast Histopathology Images (IDC)** dataset from Kaggle.

[Breast Histopathology Images Dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

The dataset contains **277,524 labeled 50×50 RGB patches** from **279 patient folders**, originally extracted from **162 whole slide images**.

```text
data/
├── <patient_id>/
│   ├── 0/    # IDC negative
│   └── 1/    # IDC positive
```

The dataset is split by **patient ID** to prevent patient overlap between training, validation, and test sets.

## CNN Architecture

```text
Input 50×50×3
      ↓
Conv2D 32 + BatchNorm
      ↓
Conv2D 32 + BatchNorm
      ↓
MaxPooling
      ↓
Conv2D 64 + BatchNorm
      ↓
Conv2D 64 + BatchNorm
      ↓
MaxPooling
      ↓
Conv2D 128 + BatchNorm
      ↓
Conv2D 128 + BatchNorm
      ↓
MaxPooling
      ↓
GlobalAveragePooling2D
      ↓
Dropout 0.5
      ↓
Dense 256
      ↓
Dropout 0.3
      ↓
Sigmoid Output
```

**Total Parameters:** 322,081

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/Subinthapa2092/breast-cancer-idc-detection.git
cd breast-cancer-idc-detection
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```powershell
venv\Scripts\activate
```

### macOS and Linux

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Setup

Download the dataset using the Kaggle CLI:

```bash
kaggle datasets download -d paultimothymooney/breast-histopathology-images
```

Extract the dataset so that the patient folders are directly inside the `data` directory.

## Training

### Full Dataset

```bash
python -m src.train
```

### Faster CPU Experiment

```bash
python -m src.train --model_type scratch --max_train_samples 20000 --max_val_samples 3000 --skip_baseline
```

## Transfer Learning

### MobileNetV2

```bash
python -m src.train --model_type transfer --backbone mobilenet_v2
```

### EfficientNetB0

```bash
python -m src.train --model_type transfer --backbone efficientnet_b0
```

### ResNet50

```bash
python -m src.train --model_type transfer --backbone resnet50
```

## Evaluation

```bash
python -m src.evaluate
```

The evaluation pipeline generates ROC AUC, accuracy, precision, recall, F1 score, ROC curves, and confusion matrices in the `outputs` directory.

## Single Image Prediction

```bash
python -m src.predict --image_path data/10253/1/10253_idx5_x501_y351_class1.png
```

> **Note:** Predictions from this project are experimental and should not be interpreted as medical diagnoses.

The same prediction logic is also available through the [live web app](https://breast-cancer-idc-detection-qujh.onrender.com), which wraps this model in a FastAPI backend with a browser based upload interface.

## Project Structure

```text
breast-cancer-idc-detection/
├── data/
├── models/
├── notebooks/
├── outputs/
├── src/
│   ├── baseline.py
│   ├── data_loader.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── config.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── LICENSE
└── README.md
```

## License

**MIT License**

## Author

**Subin Thapa**

[GitHub Repository](https://github.com/Subinthapa2092/breast-cancer-idc-detection)