import os
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import nibabel as nib

from utils import preprocess_ct_image_for_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

CLASS_MAP = {
    "ADC": 1,
    "LCC": 2,
    "SCC": 3
}

DATASET_ID = 1
DATASET_NAME = "LungCancer"
DATASET_FOLDER = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"

# ============================================================
# PATH SETUP (FIXED)
# ============================================================

def get_project_root() -> Path:
    # backend/src/convert_jpg_to_nifti.py → backend/
    return Path(__file__).resolve().parents[1]


def setup_nnunet_folders():
    project_root = get_project_root()
    base_path = project_root / "data" / "nnunet"

    os.environ["nnUNet_raw"] = str(base_path / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(base_path / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(base_path / "nnUNet_results")

    for key in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
        os.makedirs(os.environ[key], exist_ok=True)

    dataset_path = Path(os.environ["nnUNet_raw"]) / DATASET_FOLDER
    (dataset_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (dataset_path / "imagesTs").mkdir(parents=True, exist_ok=True)

    return dataset_path

# ============================================================
# DATASET.JSON
# ============================================================

def write_dataset_json(dataset_path: Path, num_training: int, num_test: int):
    dataset_json = {
        "channel_names": {0: "CT"},
        "labels": {
            "background": 0,
            "ADC": 1,
            "LCC": 2,
            "SCC": 3
        },
        "numTraining": num_training,
        "numTest": num_test,
        "file_ending": ".nii.gz",
        "dataset_name": DATASET_NAME,
        "tensorImageSize": "2D"
    }

    with open(dataset_path / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    logger.info("dataset.json written successfully")

# ============================================================
# SAVE FUNCTIONS
# ============================================================

def save_ct_nifti(image: np.ndarray, output_path: Path):
    image = preprocess_ct_image_for_training(image)
    nib.save(nib.Nifti1Image(image, np.eye(4)), output_path)


def save_mask_nifti(mask: np.ndarray, class_id: int, output_path: Path):
    binary = (mask > 0).astype(np.uint8)
    label = binary * class_id
    nib.save(nib.Nifti1Image(label, np.eye(4)), output_path)

# ============================================================
# CORE CONVERSION
# ============================================================

def get_image_files(folder: Path):
    return (
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.JPG")) +
        list(folder.glob("*.png")) +
        list(folder.glob("*.jpeg"))
    )


def convert_split(split_dir: Path, dataset_path: Path, is_train: bool):
    ct_root = split_dir / "CT"
    mask_root = split_dir / "MASK"

    images_dir = dataset_path / ("imagesTr" if is_train else "imagesTs")
    labels_dir = dataset_path / "labelsTr"

    index = 1

    for class_name, class_id in CLASS_MAP.items():
        ct_class_dir = ct_root / class_name
        mask_class_dir = mask_root / class_name

        logger.info(f"Scanning: {ct_class_dir}")

        if not ct_class_dir.exists():
            logger.warning(f"Missing folder: {ct_class_dir}")
            continue

        image_files = get_image_files(ct_class_dir)

        for img_file in sorted(image_files):
            base_name = f"{DATASET_NAME}_{index:04d}"

            ct_img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if ct_img is None:
                continue

            save_ct_nifti(
                ct_img,
                images_dir / f"{base_name}_0000.nii.gz"
            )

            if is_train:
                mask_file = mask_class_dir / img_file.name
                if not mask_file.exists():
                    logger.warning(f"Missing mask: {mask_file}")
                    continue

                mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    continue

                save_mask_nifti(
                    mask_img,
                    class_id,
                    labels_dir / f"{base_name}.nii.gz"
                )

            index += 1

    return index - 1

# ============================================================
# MAIN
# ============================================================

def run_full_conversion():
    project_root = get_project_root()
    data_root = project_root / "data"

    dataset_path = setup_nnunet_folders()

    logger.info("Converting TRAIN data...")
    num_train = convert_split(data_root / "raw" / "train", dataset_path, True)

    logger.info("Converting TEST data...")
    num_test = convert_split(data_root / "raw" / "test", dataset_path, False)

    write_dataset_json(dataset_path, num_train, num_test)

    logger.info("✅ JPG → NIfTI conversion completed")
    logger.info(f"Training samples: {num_train}")
    logger.info(f"Test samples: {num_test}")

if __name__ == "__main__":
    run_full_conversion()
