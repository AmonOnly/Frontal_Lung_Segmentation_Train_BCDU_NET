# Frontal Lung Segmentation with BCDU-Net

This repository contains an adaptation of the BCDU-Net architecture for frontal chest X-ray lung segmentation in the context of a Computer Science undergraduate thesis project.

## Overview

The original BCDU-Net was introduced by Azad et al. for medical image segmentation and is available in the original repositories:

- Original BCDU-Net repository: https://github.com/rezazad68/BCDU-Net
- Related LSTM-U-Net repository: https://github.com/rezazad68/LSTM-U-net
- Original paper: Azad et al., "Bi-Directional ConvLSTM U-Net with Densely Connected Convolutions", ICCV Workshops, 2019

This project does not replace the original authors' work. It is a pipeline adaptation for a different use case: frontal chest radiographs and pulmonary mask extraction for later ROI isolation in a TCC workflow.

The final goal is to segment lung fields from frontal AP/PA chest radiographs and use the resulting binary masks to isolate the pulmonary region before downstream similarity analysis.

## Project objective

This repository focuses on:

- frontal chest X-ray images
- grayscale conversion
- resizing to 256x256
- lung mask generation
- BCDU-Net D3 training
- validation on a held-out test set
- binary lung segmentation for later ROI-based analysis

This is not a diagnostic pipeline or a disease classification model. It is a segmentation step used to obtain pulmonary masks.

## Architecture

The model used in this project is the BCDU-Net D3 variant:

- input: 256 x 256 x 1
- dense convolutions
- ConvLSTM2D in skip connections
- BatchNormalization
- decoder with transposed convolutions
- final output: 1 channel
- activation: sigmoid

The model is kept aligned with the original BCDU-Net concept and is not replaced by a generic U-Net.

## Repository structure

```text
Frontal_Lung_Segmentation_Train_BCDU_NET/
├── README.md
├── .gitignore
├── requirements.txt
├── BCDU-Net_MSc-master/
│   └── Lung Segmentation/
│       ├── Prepare_data.py
│       ├── train_lung.py
│       ├── models.py
│       ├── evaluate_performance.py
│       ├── Reza_functions.py
│       └── processed_data/   (generated locally, not committed)
└── 1/
    └── data/
        └── Lung Segmentation/
```

## Dataset

The segmentation dataset used in this project is a frontal chest X-ray dataset with lung masks, based on the Kaggle chest X-ray masks and labels resource:

- Kaggle dataset: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
- Reference notebook: https://www.kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset

The dataset should be organized in a structure compatible with the current preprocessing script.

### Data organization expected by Prepare_data.py

```text
1/
└── data/
    └── Lung Segmentation/
        ├── CXR_png/
        ├── masks/
        ├── ClinicalReadings/
        └── test/
```

The raw dataset itself is intentionally not committed to GitHub.

## CheXpert stage

CheXpert is used later in the TCC for inference and downstream analysis after the lung segmentation model is trained.

The CheXpert directory is not included in this repository and should be kept outside the GitHub project.

Expected later structure (conceptual only):

```text
Base_CheXpert_IC/
├── valid_desconhecidos_frontal/
├── valid_Doentes_frontal/
└── valid_normais_frontal/
```

This stage belongs to the post-training inferential evaluation of the TCC and is not part of the repository publication.

## Dependencies

The project uses:

- Python 3
- TensorFlow
- Keras
- NumPy
- SciPy
- scikit-learn
- matplotlib
- Pillow
- pandas

A compatible environment should be used. The project was validated in a Python environment with a working TensorFlow/Keras stack.

## Data preparation

The preprocessing script adapts the original CT-based pipeline to a frontal CXR workflow.

Run:

```bash
cd "BCDU-Net_MSc-master/Lung Segmentation"
python3 Prepare_data.py
```

This script:

- reads CXR images and corresponding masks
- matches image/mask pairs
- converts X-rays to grayscale
- resizes to 256x256
- uses nearest-neighbor interpolation for masks
- binarizes masks
- removes missing pairs
- splits by patient to avoid leakage
- uses seed 42
- saves arrays in processed_data/
- saves splits.csv
- generates visual validation examples

Important: the legacy CT-specific logic is not used here. Items such as nibabel, HU, FOV, around_lung, and return_axials are not part of the frontal CXR pipeline.

## Training

Run:

```bash
cd "BCDU-Net_MSc-master/Lung Segmentation"
python3 train_lung.py
```

Main parameters used:

- input size: 256 x 256 x 1
- batch size: 2
- epochs: 50
- optimizer: Adam
- learning rate: 1e-4
- loss: binary crossentropy
- metrics: accuracy, Dice, IoU, precision, recall

Callbacks:

- ModelCheckpoint
- ReduceLROnPlateau
- EarlyStopping
- CSVLogger

The best model is selected according to validation loss.

## Evaluation

Run:

```bash
cd "BCDU-Net_MSc-master/Lung Segmentation"
python3 evaluate_performance.py
```

The evaluation script uses the isolated test set only and computes:

- Accuracy
- Dice
- IoU
- Precision
- Recall
- Specificity
- AUC

The predictions are generated from the model probability map with a threshold of 0.5.

## Test set results

The following metrics correspond to the isolated test set used for validation of the trained lung segmentation model:

| Metric | Value |
|---|---:|
| Accuracy | 0.9831488779 |
| Dice | 0.9659835903 |
| IoU | 0.9342052807 |
| Precision | 0.9740325937 |
| Recall | 0.9580665239 |
| Specificity | 0.9914979789 |
| AUC | 0.9977695376 |

These results refer to the manually annotated test split used for evaluation in this project and should not be interpreted as CheXpert diagnostic results.

## Pre-trained weights

The project can use a best checkpoint named:

```text
BCDU-Net_MSc-master/Lung Segmentation/models/weights/bcdu_cxr/best_bcdu_cxr.weights.h5
```

This file is not included in the repository because it is large and is generated by the training workflow. The user should place it in the expected path if they want to run evaluation or inference on the saved model.

## Files intentionally excluded from GitHub

The following are intentionally not published with the repository:

- raw datasets
- medical imaging folders
- processed arrays (.npy)
- model checkpoints (.weights.h5, .h5, .keras)
- virtual environments (.venv)
- logs and training artifacts
- generated results folders
- archives such as .zip, .tar.gz
- Kaggle credentials or local environment files

This keeps the repository focused on code, documentation, and reproducible instructions.

## Original BCDU-Net attribution

This project is inspired by and adapted from the original BCDU-Net work by the original authors:

- R. Azad
- M. Asadi
- Mahmood Fathy
- Sergio Escalera

Reference:

> Azad, R., et al. "Bi-Directional ConvLSTM U-Net with Densely Connected Convolutions". ICCV Workshops, 2019.

The project preserves the original authors' attribution and acknowledges that the architecture is derived from their work while adapting the data pipeline to frontal chest X-ray lung segmentation.

## License and provenance

This repository keeps the original attribution and does not claim authorship of the original BCDU-Net implementation. The adaptation is made for academic research and reproducibility within this TCC project.

## Citation

If the original architecture is used as part of a publication, please cite the original BCDU-Net paper:

```bibtex
@inproceedings{azad2019bcdunet,
  title={Bi-Directional ConvLSTM U-Net with Densely Connected Convolutions},
  author={Azad, R. and Asadi, M. and Fathy, M. and Escalera, S.},
  booktitle={ICCV Workshops},
  year={2019}
}
```

## Notes

This repository is designed to be reproducible and publication-ready on GitHub without uploading patient data, generated arrays, or large checkpoint files.

The code is ready to be cloned and run by installing dependencies, downloading the public chest X-ray segmentation dataset, preparing the data, training the model, and evaluating the isolated test split.
