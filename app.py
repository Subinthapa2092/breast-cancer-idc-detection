"""
FastAPI web app for the IDC breast cancer CNN.

Serves a single-page frontend (templates/index.html) and an inference
endpoint that runs the trained CNN on an uploaded patch image.

Run with:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000 in a browser.
"""

import io

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import config

app = FastAPI(title="IDC Breast Cancer Detection")

app.mount("/static", StaticFiles(directory="static"), name="static")

_model = None


def get_model():
    global _model
    if _model is None:
        if not __import__("os").path.exists(config.CNN_MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No trained model found at {config.CNN_MODEL_PATH}. "
                    "Train one first with: python -m src.train"
                ),
            )
        _model = tf.keras.models.load_model(config.CNN_MODEL_PATH)
    return _model


@app.get("/", response_class=HTMLResponse)
def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    model = get_model()

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that image file.")

    img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0).ravel()[0])
    label = 1 if prob > 0.5 else 0

    return JSONResponse(
        {
            "probability": prob,
            "label": label,
            "label_name": config.LABEL_NAMES[label],
        }
    )


@app.get("/api/health")
def health():
    import os

    return {
        "model_available": os.path.exists(config.CNN_MODEL_PATH),
        "model_path": config.CNN_MODEL_PATH,
    }
