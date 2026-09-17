"""Prepare frontal chest radiographs and lung masks for BCDU-Net."""

from pathlib import Path
import csv
import re

import numpy as np
from PIL import Image


IMAGE_SIZE = 256
SEED = 42
MASK_THRESHOLD = 127

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parents[1] / "1" / "data" / "Lung Segmentation"
OUTPUT_DIR = SCRIPT_DIR / "processed_data"
VISUAL_DIR = SCRIPT_DIR / "results" / "dataset_validation"


def image_key(path):
    name = path.stem
    if name.endswith("_mask"):
        name = name[:-5]
    return name


def patient_id(key):
    return re.sub(r"_[0-9]+$", "", key)


def index_files(directory):
    indexed = {}
    duplicates = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        key = image_key(path)
        if key in indexed:
            duplicates.append(str(path))
        else:
            indexed[key] = path
    return indexed, duplicates


def load_pair(image_path, mask_path):
    with Image.open(image_path) as image_file:
        image = np.asarray(image_file.convert("L").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)
    with Image.open(mask_path) as mask_file:
        mask = np.asarray(mask_file.convert("L").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST), dtype=np.uint8)
    mask = (mask > MASK_THRESHOLD).astype(np.float32)
    return image, mask


def save_visual_checks(pairs, images, masks):
    import matplotlib.pyplot as plt

    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    for index in range(min(5, len(pairs))):
        _, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(images[index], cmap="gray")
        axes[0].set_title("Radiograph")
        axes[1].imshow(masks[index], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Mask")
        axes[2].imshow(images[index], cmap="gray")
        axes[2].imshow(masks[index], cmap="Reds", alpha=0.35, vmin=0, vmax=1)
        axes[2].set_title("Overlay")
        for axis in axes:
            axis.axis("off")
        plt.tight_layout()
        plt.savefig(VISUAL_DIR / (pairs[index][0].stem + "_check.png"), dpi=120)
        plt.close()


def main():
    image_dir = DATASET_DIR / "CXR_png"
    mask_dir = DATASET_DIR / "masks"
    image_files, image_duplicates = index_files(image_dir)
    mask_files, mask_duplicates = index_files(mask_dir)
    image_keys = set(image_files)
    mask_keys = set(mask_files)
    valid_keys = sorted(image_keys & mask_keys)
    missing_masks = sorted(image_keys - mask_keys)
    missing_images = sorted(mask_keys - image_keys)

    print("=" * 50)
    print("CXR DATA PREPARATION")
    print("=" * 50)
    print("Images found:", len(image_files))
    print("Masks found:", len(mask_files))
    print("Valid pairs:", len(valid_keys))
    print("Image extensions:", sorted({path.suffix.lower() for path in image_files.values()}))
    print("Mask extensions:", sorted({path.suffix.lower() for path in mask_files.values()}))
    print("Images without masks:", missing_masks)
    print("Masks without images:", missing_images)
    print("Duplicate image keys:", image_duplicates)
    print("Duplicate mask keys:", mask_duplicates)
    if not valid_keys:
        raise RuntimeError("No valid image-mask pairs found.")

    pairs = [(image_files[key], mask_files[key], patient_id(key)) for key in valid_keys]
    valid_pairs, images, masks = [], [], []
    for image_path, mask_path, _ in pairs:
        try:
            image, mask = load_pair(image_path, mask_path)
        except Exception as error:
            print("Corrupt pair skipped:", image_path.name, mask_path.name, error)
            continue
        valid_pairs.append((image_path, mask_path, patient_id(image_key(image_path))))
        images.append(image)
        masks.append(mask)
    pairs = valid_pairs
    images = np.asarray(images, dtype=np.uint8)
    masks = np.asarray(masks, dtype=np.float32)
    if len(pairs) != len(images):
        raise RuntimeError("Internal pair bookkeeping error.")

    rng = np.random.default_rng(SEED)
    patients = np.array(sorted({pair[2] for pair in pairs}))
    rng.shuffle(patients)
    train_end = int(len(patients) * 0.70)
    val_end = train_end + int(len(patients) * 0.15)
    split_by_patient = {
        patient: "train" if index < train_end else "val" if index < val_end else "test"
        for index, patient in enumerate(patients)
    }
    split_names = np.array([split_by_patient[pair[2]] for pair in pairs])
    indices = {name: np.where(split_names == name)[0] for name in ("train", "val", "test")}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        np.save(OUTPUT_DIR / ("data_" + split), images[indices[split]])
        np.save(OUTPUT_DIR / ("mask_" + split), masks[indices[split]])
    with (OUTPUT_DIR / "splits.csv").open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image", "mask", "patient_id", "split"])
        for pair, split in zip(pairs, split_names):
            writer.writerow([pair[0].name, pair[1].name, pair[2], split])

    save_visual_checks(pairs, images, masks)
    print("Target size:", str(IMAGE_SIZE) + "x" + str(IMAGE_SIZE))
    print("Train:", len(indices["train"]), "Validation:", len(indices["val"]), "Test:", len(indices["test"]))
    print("X_train:", images[indices["train"]].shape)
    print("Y_train:", masks[indices["train"]].shape)
    print("Masks unique values:", np.unique(masks))
    print("Saved:", OUTPUT_DIR)
    print("STATUS: PASS")


if __name__ == "__main__":
    main()


