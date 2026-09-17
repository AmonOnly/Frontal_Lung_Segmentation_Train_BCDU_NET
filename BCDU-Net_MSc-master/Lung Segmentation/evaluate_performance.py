"""Evaluate BCDU-Net on the isolated frontal CXR test set."""

from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, jaccard_score, precision_score, recall_score, roc_auc_score

import models as M

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "processed_data"
WEIGHTS = SCRIPT_DIR / "models" / "weights" / "bcdu_cxr" / "best_bcdu_cxr.weights.h5"
OUTPUT_DIR = SCRIPT_DIR / "results" / "evaluation" / "bcdu_cxr"


def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection + 1e-7) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)


def main():
    test_data = np.load(DATA_DIR / "data_test.npy").astype(np.float32) / 255.0
    test_mask = np.load(DATA_DIR / "mask_test.npy").astype(np.float32)
    test_data = np.expand_dims(test_data, axis=3)
    model = M.BCDU_net_D3(input_size=(256, 256, 1))
    model.load_weights(WEIGHTS)
    probabilities = model.predict(test_data, batch_size=2, verbose=1)
    predictions = (probabilities > 0.5).astype(np.float32)
    y_true = test_mask.reshape(-1).astype(np.uint8)
    y_pred = predictions.reshape(-1).astype(np.uint8)
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()
    metrics = {
        "accuracy": (tn + tp) / max(1, tn + fp + fn + tp),
        "dice": dice_score(y_true, y_pred),
        "iou": jaccard_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / max(1, tn + fp),
        "auc": roc_auc_score(y_true, probabilities.reshape(-1)),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "metrics.txt").open("w") as output:
        output.write("Confusion matrix:\n" + str(confusion) + "\n")
        for name, value in metrics.items():
            output.write(name + ": " + str(value) + "\n")
    np.save(OUTPUT_DIR / "test_probabilities.npy", probabilities)
    np.save(OUTPUT_DIR / "test_predictions.npy", predictions)
    print("Test samples:", len(test_data))
    print("Probabilities shape:", probabilities.shape)
    print("Predicted mask values:", np.unique(predictions))
    print("Confusion matrix:\n", confusion)
    for name, value in metrics.items():
        print(name + ":", value)
    print("Saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
