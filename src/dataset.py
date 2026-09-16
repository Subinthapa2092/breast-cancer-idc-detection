"""
tf.data pipeline for the IDC breast histopathology patches.

Unlike the earlier PatchCamelyon project (which stored .tif patches and
needed a PIL/tf.py_function workaround), this dataset's patches are .png,
which tf.io.decode_png handles natively. No py_function needed.
"""

import tensorflow as tf


def _load_image(filepath, label, image_size):
    img_bytes = tf.io.read_file(filepath)
    img = tf.io.decode_png(img_bytes, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32)
    return img, label


def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    return img, label


def make_dataset(
    df,
    image_size: int,
    batch_size: int,
    training: bool = False,
    shuffle_buffer: int = 4096,
):
    """Build a tf.data.Dataset from a dataframe with 'filepath' and 'label'."""
    filepaths = df["filepath"].values
    labels = df["label"].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if training:
        ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(df)), seed=42)

    ds = ds.map(
        lambda fp, lb: _load_image(fp, lb, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
