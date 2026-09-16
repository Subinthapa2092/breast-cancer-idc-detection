"""
Train the baseline and/or CNN on the IDC breast histopathology dataset.

Examples
--------
Baseline + from-scratch CNN, full dataset:
    python -m src.train

Fast CPU test on a subsample, skipping the baseline:
    python -m src.train --model_type scratch --max_train_samples 20000 \
        --max_val_samples 3000 --skip_baseline

Transfer learning with MobileNetV2:
    python -m src.train --model_type transfer --backbone mobilenet_v2 \
        --max_train_samples 20000 --max_val_samples 3000 --skip_baseline
"""

import argparse

import tensorflow as tf

import config
from src.data_loader import load_labels
from src.preprocessing import make_patient_level_splits, compute_class_weights
from src.dataset import make_dataset
from src.model import build_cnn_from_scratch, build_transfer_model, compile_model
from src.baseline import train_baseline


def parse_args():
    parser = argparse.ArgumentParser(description="Train IDC breast cancer classifier")
    parser.add_argument(
        "--model_type", choices=["scratch", "transfer"], default="scratch"
    )
    parser.add_argument(
        "--backbone",
        choices=["mobilenet_v2", "efficientnet_b0", "resnet50"],
        default="mobilenet_v2",
        help="Only used when --model_type transfer",
    )
    parser.add_argument("--skip_cnn", action="store_true")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="If set, randomly subsample the training set to this many rows (fast CPU testing).",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="If set, randomly subsample the validation set to this many rows.",
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading labels...")
    df = load_labels(config.DATA_DIR)
    print(f"Total patches: {len(df):,} from {df['patient_id'].nunique():,} patients")

    train_df, val_df, test_df = make_patient_level_splits(
        df, config.VAL_SIZE, config.TEST_SIZE, config.RANDOM_STATE
    )

    if args.max_train_samples and len(train_df) > args.max_train_samples:
        train_df = train_df.sample(
            n=args.max_train_samples, random_state=config.RANDOM_STATE
        ).reset_index(drop=True)
    if args.max_val_samples and len(val_df) > args.max_val_samples:
        val_df = val_df.sample(
            n=args.max_val_samples, random_state=config.RANDOM_STATE
        ).reset_index(drop=True)

    print(
        f"Using {len(train_df):,} train / {len(val_df):,} val / "
        f"{len(test_df):,} test rows for this run."
    )

    # --- Baseline -------------------------------------------------------
    if not args.skip_baseline:
        print("\n=== Training baseline (color histogram + logistic regression) ===")
        train_baseline(train_df, val_df, config.BASELINE_MODEL_PATH)
    else:
        print("\nSkipping baseline (--skip_baseline)")

    # --- CNN --------------------------------------------------------------
    if args.skip_cnn:
        print("\nSkipping CNN (--skip_cnn)")
        return

    print(f"\n=== Training CNN ({args.model_type}) ===")

    train_ds = make_dataset(
        train_df, config.IMAGE_SIZE, args.batch_size, training=True
    )
    val_ds = make_dataset(val_df, config.IMAGE_SIZE, args.batch_size, training=False)

    if args.model_type == "scratch":
        model = build_cnn_from_scratch(config.IMAGE_SIZE, config.CHANNELS)
        save_path = config.CNN_MODEL_PATH
    else:
        model = build_transfer_model(
            args.backbone, config.IMAGE_SIZE, config.CHANNELS
        )
        save_path = config.CNN_MODEL_PATH.replace(".keras", f"_{args.backbone}.keras")

    model = compile_model(model, config.LEARNING_RATE)
    model.summary()

    class_weights = compute_class_weights(train_df)
    print(f"Class weights: {class_weights}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save(save_path)
    print(f"Saved trained CNN to {save_path}")


if __name__ == "__main__":
    main()
