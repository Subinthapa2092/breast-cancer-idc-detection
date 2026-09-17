# """
# FastAPI web app for the IDC breast cancer CNN.

# Serves a single-page frontend (templates/index.html) and an inference
# endpoint that runs the trained CNN on an uploaded patch image.

# Run with:
#     uvicorn app:app --reload

# Then open http://127.0.0.1:8000 in a browser.
# """

# import io

# import numpy as np
# import tensorflow as tf
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from PIL import Image

# import config

# app = FastAPI(title="IDC Breast Cancer Detection")

# app.mount("/static", StaticFiles(directory="static"), name="static")

# _model = None


# def get_model():
#     global _model
#     if _model is None:
#         if not __import__("os").path.exists(config.CNN_MODEL_PATH):
#             raise HTTPException(
#                 status_code=503,
#                 detail=(
#                     f"No trained model found at {config.CNN_MODEL_PATH}. "
#                     "Train one first with: python -m src.train"
#                 ),
#             )
#         _model = tf.keras.models.load_model(config.CNN_MODEL_PATH)
#     return _model


# @app.get("/", response_class=HTMLResponse)
# def index():
#     with open("templates/index.html", encoding="utf-8") as f:
#         return f.read()


# @app.post("/api/predict")
# async def predict(file: UploadFile = File(...)):
#     if not file.content_type or not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="Please upload an image file.")

#     model = get_model()

#     raw = await file.read()
#     try:
#         img = Image.open(io.BytesIO(raw)).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=400, detail="Could not read that image file.")

#     img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
#     arr = np.asarray(img, dtype=np.float32)
#     arr = np.expand_dims(arr, axis=0)

#     prob = float(model.predict(arr, verbose=0).ravel()[0])
#     label = 1 if prob > 0.5 else 0

#     return JSONResponse(
#         {
#             "probability": prob,
#             "label": label,
#             "label_name": config.LABEL_NAMES[label],
#         }
#     )


# @app.get("/api/health")
# def health():
#     import os

#     return {
#         "model_available": os.path.exists(config.CNN_MODEL_PATH),
#         "model_path": config.CNN_MODEL_PATH,
#     }
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


def check_image_plausibility(img: Image.Image) -> str | None:
    """Lightweight heuristic check for whether an uploaded image plausibly
    resembles an H and E stained tissue patch, as opposed to an unrelated
    photo (a document, a screenshot, a random object, and so on).

    This is NOT a learned out of distribution detector, just simple color
    statistics. Real tissue patches are densely colored across the whole
    frame, with visible pink or purple staining from the H and E process.
    Unrelated images, such as scanned text or plain photos, are often
    mostly desaturated, or mostly a single flat background color.

    Returns a short warning string if the image looks unlikely to be a
    valid patch, or None if it looks plausible. The prediction still runs
    either way. This only adds a caution note, it does not block anything.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    saturation = np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-6, None), 0)
    mean_saturation = float(saturation.mean())

    near_white_fraction = float(((r > 0.85) & (g > 0.85) & (b > 0.85)).mean())

    if mean_saturation < 0.08:
        return (
            "This image looks mostly grayscale. Tissue patches usually show "
            "visible pink or purple staining, so this result may not be "
            "meaningful."
        )

    if near_white_fraction > 0.85:
        return (
            "This image is almost entirely plain background. Tissue patches "
            "are typically colored across the whole frame, so this result "
            "may not be meaningful."
        )

    return None


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

    warning = check_image_plausibility(img)

    resized = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    arr = np.asarray(resized, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0).ravel()[0])
    label = 1 if prob > 0.5 else 0

    return JSONResponse(
        {
            "probability": prob,
            "label": label,
            "label_name": config.LABEL_NAMES[label],
            "warning": warning,
        }
    )


@app.get("/api/health")
def health():
    import os

    return {
        "model_available": os.path.exists(config.CNN_MODEL_PATH),
        "model_path": config.CNN_MODEL_PATH,
    }
