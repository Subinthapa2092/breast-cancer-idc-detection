"""
Run the trained CNN on a single patch image.

Usage:
    python -m src.predict --image_path data/<patient_id>/1/<patch>.png
"""

import argparse

import numpy as np
import tensorflow as tf

import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model_path", default=config.CNN_MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    model = tf.keras.models.load_model(args.model_path)

    img_bytes = tf.io.read_file(args.image_path)
    img = tf.io.decode_png(img_bytes, channels=3)
    img = tf.image.resize(img, [config.IMAGE_SIZE, config.IMAGE_SIZE])
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)

    prob = float(model.predict(img).ravel()[0])
    label = 1 if prob > 0.5 else 0

    print(f"Image: {args.image_path}")
    print(f"Predicted probability of IDC positive: {prob:.4f}")
    print(f"Predicted label: {label} ({config.LABEL_NAMES[label]})")


if __name__ == "__main__":
    main()
