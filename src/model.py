"""
Model architectures for the IDC breast cancer patch classifier.

The from-scratch CNN uses 3 conv blocks instead of 4. With a 50x50 input,
four rounds of /2 max-pooling shrinks the feature map to ~3px, which is too
aggressive and throws away spatial information; three blocks bottoms out
at a more reasonable ~6px feature map before global average pooling.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_from_scratch(image_size: int = 50, channels: int = 3) -> tf.keras.Model:
    inputs = layers.Input(shape=(image_size, image_size, channels))
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="idc_cnn_from_scratch")


_BACKBONES = {
    "mobilenet_v2": tf.keras.applications.MobileNetV2,
    "efficientnet_b0": tf.keras.applications.EfficientNetB0,
    "resnet50": tf.keras.applications.ResNet50,
}


def build_transfer_model(
    backbone_name: str, image_size: int = 50, channels: int = 3, trainable: bool = False
) -> tf.keras.Model:
    if backbone_name not in _BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. Choose from {list(_BACKBONES)}"
        )

    # Pretrained backbones expect >= 32x32 (MobileNetV2 wants >= 96 to be
    # safe with its stride stack); IDC patches are 50x50, which works but
    # we resize up slightly for the classic ImageNet backbones' stability.
    input_size = max(image_size, 96)

    inputs = layers.Input(shape=(image_size, image_size, channels))
    x = layers.Resizing(input_size, input_size)(inputs)

    backbone_cls = _BACKBONES[backbone_name]
    base_model = backbone_cls(
        include_top=False, weights="imagenet", input_shape=(input_size, input_size, 3)
    )
    base_model.trainable = trainable

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    if backbone_name == "efficientnet_b0":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    elif backbone_name == "resnet50":
        preprocess = tf.keras.applications.resnet50.preprocess_input

    x = preprocess(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs, name=f"idc_transfer_{backbone_name}")


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
