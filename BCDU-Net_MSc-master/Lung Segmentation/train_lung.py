"""Train BCDU-Net D3 on frontal chest radiographs."""

from pathlib import Path
import csv
import os

import numpy as np
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import Precision, Recall
from keras import ops

import models as M


IMAGE_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 50
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "processed_data"
MODEL_DIR = SCRIPT_DIR / "models" / "weights" / "bcdu_cxr"
RESULTS_DIR = SCRIPT_DIR / "results" / "training" / "bcdu_cxr"


def get_last_epoch(history_path):
        if not history_path.exists():
                return 0
        with history_path.open() as csv_file:
                rows = [line.strip() for line in csv_file if line.strip()]
        if len(rows) <= 1:
                return 0
        try:
                return int(rows[-1].split(',')[0]) + 1
        except (ValueError, IndexError):
                return 0


def dice_metric(y_true, y_pred):
        y_true = ops.reshape(y_true, (-1,))
        y_pred = ops.reshape(y_pred, (-1,))
        intersection = ops.sum(y_true * y_pred)
        epsilon = 1e-7
        return (2.0 * intersection + epsilon) / (ops.sum(y_true) + ops.sum(y_pred) + epsilon)


def iou_metric(y_true, y_pred):
        y_true = ops.reshape(y_true, (-1,))
        y_pred = ops.reshape(y_pred, (-1,))
        intersection = ops.sum(y_true * y_pred)
        union = ops.sum(y_true) + ops.sum(y_pred) - intersection
        epsilon = 1e-7
        return (intersection + epsilon) / (union + epsilon)


def main():
        train_data = np.load(DATA_DIR / "data_train.npy").astype(np.float32) / 255.0
        train_mask = np.load(DATA_DIR / "mask_train.npy").astype(np.float32)
        val_data = np.load(DATA_DIR / "data_val.npy").astype(np.float32) / 255.0
        val_mask = np.load(DATA_DIR / "mask_val.npy").astype(np.float32)
        train_data = np.expand_dims(train_data, axis=3)
        train_mask = np.expand_dims(train_mask, axis=3)
        val_data = np.expand_dims(val_data, axis=3)
        val_mask = np.expand_dims(val_mask, axis=3)
        expected_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)
        for name, array in (("X_train", train_data), ("Y_train", train_mask), ("X_val", val_data), ("Y_val", val_mask)):
                if array.shape[1:] != expected_shape:
                        raise ValueError(name + " has unexpected shape " + str(array.shape))
                print(name + ":", array.shape, array.dtype)
        if not np.isfinite(train_data).all() or not np.isfinite(val_data).all():
                raise ValueError("Image data contains NaN or Infinity.")
        if not np.isin(train_mask, [0.0, 1.0]).all() or not np.isin(val_mask, [0.0, 1.0]).all():
                raise ValueError("Masks must contain only 0 and 1.")

        print("=" * 50)
        print("BCDU-NET CXR LUNG SEGMENTATION")
        print("=" * 50)
        print("Training samples:", len(train_data))
        print("Validation samples:", len(val_data))
        print("Input: 256x256x1")
        print("Model: BCDU_net_D3")
        print("Optimizer: Adam")
        print("Learning rate: 0.0001")
        print("Batch size:", BATCH_SIZE)
        smoke_test = os.environ.get("SMOKE_TEST", "0") == "1"
        epochs = 1 if smoke_test else EPOCHS
        print("Epochs:", epochs)

        model = M.BCDU_net_D3(input_size=expected_shape)
        model.load_weights(str(MODEL_DIR / "best_bcdu_cxr.weights.h5"))
        model.compile(optimizer=model.optimizer, loss="binary_crossentropy",
                                  metrics=["accuracy", dice_metric, iou_metric, Precision(name="precision"), Recall(name="recall")])
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = ModelCheckpoint(MODEL_DIR / "best_bcdu_cxr.weights.h5", save_best_only=True,
                                                                 save_weights_only=True, monitor="val_loss", mode="min")
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=7, mode="min", verbose=1)
        early_stopping = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, mode="min")
        history_path = RESULTS_DIR / "history.csv"
        initial_epoch = get_last_epoch(history_path)
        if initial_epoch > 0:
                print("Resuming from epoch:", initial_epoch)
        csv_logger = CSVLogger(history_path, append=True)
        print("Training...")
        fit_options = {}
        if smoke_test:
                fit_options = {"steps_per_epoch": min(2, max(1, len(train_data) // BATCH_SIZE)), "validation_steps": 1}
        history = model.fit(train_data, train_mask, batch_size=BATCH_SIZE, epochs=epochs, initial_epoch=initial_epoch, shuffle=True,
                                                validation_data=(val_data, val_mask), callbacks=[checkpoint, reduce_lr, early_stopping, csv_logger],
                                                **fit_options)
        with (RESULTS_DIR / "history_metrics.csv").open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["epoch", "loss", "val_loss", "accuracy", "val_accuracy", "dice", "val_dice", "iou", "val_iou", "learning_rate"])
                for index, loss in enumerate(history.history["loss"]):
                        writer.writerow([index + 1, loss, history.history["val_loss"][index],
                                                         history.history["accuracy"][index], history.history["val_accuracy"][index],
                                                         history.history["dice_metric"][index], history.history["val_dice_metric"][index],
                                                         history.history["iou_metric"][index], history.history["val_iou_metric"][index],
                                                         history.history.get("learning_rate", [""] * len(history.history["loss"]))[index]])
        print("Training history saved:", RESULTS_DIR / "history.csv")


if __name__ == "__main__":
        main()



