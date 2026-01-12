# import os
# import tempfile
# import numpy as np
# import cv2
# from PIL import Image
# import torch
# import logging
# from model_loader import ModelLoader
# from convert_jpg_to_nifti import convert_jpg_to_nifti
# from utils import create_color_mask

# logger = logging.getLogger(__name__)

# class LungCancerSegmentation:
#     def __init__(self):
#         """Initialize the lung cancer segmentation pipeline"""
#         self.model_loader = ModelLoader()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         logger.info(f"Using device: {self.device}")
        
#         # Class labels
#         self.class_labels = {
#             0: "Background",
#             1: "Adenocarcinoma (ADC)",
#             2: "Large Cell Carcinoma (LCC)", 
#             3: "Squamous Cell Carcinoma (SCC)"
#         }
        
#         # Color map for visualization
#         self.color_map = {
#             0: [0, 0, 0],        # Black - Background
#             1: [255, 0, 0],      # Red - Adenocarcinoma
#             2: [0, 255, 0],      # Green - Large Cell Carcinoma
#             3: [0, 0, 255]       # Blue - Squamous Cell Carcinoma
#         }
    
#     def preprocess_image(self, image_path):
#         """
#         Preprocess JPG image for nnU-Net inference
#         """
#         try:
#             # Convert JPG to NIfTI format
#             nifti_path = convert_jpg_to_nifti(image_path)
#             return nifti_path
#         except Exception as e:
#             logger.error(f"Preprocessing failed: {e}")
#             raise
    
#     def run_inference(self, nifti_path):
#         """
#         Run nnU-Net inference on preprocessed image
#         """
#         try:
#             # Load and run model
#             prediction = self.model_loader.predict(nifti_path)
#             return prediction
#         except Exception as e:
#             logger.error(f"Inference failed: {e}")
#             # Return dummy prediction for development
#             return self._create_dummy_prediction(nifti_path)
    
#     def _create_dummy_prediction(self, nifti_path):
#         """Create dummy prediction for development/testing"""
#         logger.warning("Using dummy prediction - model not available")
        
#         # Load original image to get dimensions
#         img = cv2.imread(nifti_path.replace('.nii.gz', '.jpg'))
#         if img is None:
#             # Fallback dimensions
#             height, width = 512, 512
#         else:
#             height, width = img.shape[:2]
        
#         # Create synthetic segmentation mask
#         mask = np.zeros((height, width), dtype=np.uint8)
        
#         # Add some synthetic tumor regions
#         center_x, center_y = width // 2, height // 2
        
#         # Adenocarcinoma region (class 1)
#         cv2.circle(mask, (center_x - 50, center_y - 30), 25, 1, -1)
        
#         # Large Cell Carcinoma region (class 2)  
#         cv2.circle(mask, (center_x + 40, center_y + 20), 20, 2, -1)
        
#         # Squamous Cell Carcinoma region (class 3)
#         cv2.circle(mask, (center_x - 20, center_y + 50), 15, 3, -1)
        
#         return mask
    
#     def postprocess_prediction(self, prediction, original_image_path):
#         """
#         Convert prediction to color-coded visualization
#         """
#         try:
#             # Create color mask
#             color_mask = create_color_mask(prediction, self.color_map)
            
#             # Load original image
#             original = cv2.imread(original_image_path)
#             if original is None:
#                 raise ValueError("Could not load original image")
            
#             # Resize mask to match original image
#             if color_mask.shape[:2] != original.shape[:2]:
#                 color_mask = cv2.resize(color_mask, (original.shape[1], original.shape[0]))
            
#             # Blend original image with segmentation mask
#             alpha = 0.6  # Transparency
#             blended = cv2.addWeighted(original, alpha, color_mask, 1-alpha, 0)
            
#             return blended
            
#         except Exception as e:
#             logger.error(f"Postprocessing failed: {e}")
#             raise
    
#     def predict(self, image_path):
#         """
#         Complete prediction pipeline
#         """
#         try:
#             logger.info(f"Starting prediction for: {image_path}")
            
#             # Preprocess image
#             nifti_path = self.preprocess_image(image_path)
            
#             # Run inference
#             prediction = self.run_inference(nifti_path)
            
#             # Postprocess and create visualization
#             result_image = self.postprocess_prediction(prediction, image_path)
            
#             # Save result
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_output:
#                 output_path = temp_output.name
            
#             cv2.imwrite(output_path, result_image)
            
#             # Clean up temporary files
#             if os.path.exists(nifti_path):
#                 os.unlink(nifti_path)
            
#             logger.info(f"Prediction completed: {output_path}")
#             return output_path
            
#         except Exception as e:
#             logger.error(f"Prediction pipeline failed: {e}")
#             raise


import os
import subprocess
import logging
from pathlib import Path

import cv2
import numpy as np
import nibabel as nib

from utils import (
    create_color_mask,
    overlay_mask_on_image,
    normalize_for_display
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ID = 1
DATASET_NAME = "LungCancer"
DATASET_FOLDER = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
CONFIGURATION = "2d"

# Class colors (for visualization)
COLOR_MAP = {
    0: (0, 0, 0),       # Background
    1: (255, 0, 0),     # ADC - Red
    2: (0, 255, 0),     # LCC - Green
    3: (0, 0, 255)      # SCC - Blue
}

# ============================================================
# PATH SETUP
# ============================================================

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def setup_nnunet_environment():
    project_root = get_project_root()
    base_path = project_root / "data" / "nnunet"

    os.environ["nnUNet_raw"] = str(base_path / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(base_path / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(base_path / "nnUNet_results")

    for key in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
        os.makedirs(os.environ[key], exist_ok=True)

    dataset_path = Path(os.environ["nnUNet_raw"]) / DATASET_FOLDER
    images_ts = dataset_path / "imagesTs"
    images_ts.mkdir(parents=True, exist_ok=True)

    return dataset_path, images_ts


# ============================================================
# IMAGE PREPARATION
# ============================================================

def jpg_to_nifti_for_inference(jpg_path: Path, output_nifti: Path):
    image = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {jpg_path}")

    image = image.astype(np.float32)
    nifti = nib.Nifti1Image(image, np.eye(4))
    nib.save(nifti, output_nifti)


# ============================================================
# NNUNET ENSEMBLE INFERENCE
# ============================================================

def run_ensemble_inference():
    """
    Run nnU-Net v2 ensemble prediction (folds 0–4)
    """
    command = [
        "nnUNetv2_predict",
        "-d", str(DATASET_ID),
        "-c", CONFIGURATION,
        "-i", os.path.join(os.environ["nnUNet_raw"], DATASET_FOLDER, "imagesTs"),
        "-o", os.path.join(os.environ["nnUNet_results"], DATASET_FOLDER, "predictions"),
        "--save_probabilities"
    ]

    logger.info("▶ Running nnU-Net v2 ENSEMBLE inference")
    logger.info(" ".join(command))

    subprocess.run(command, check=True)
    logger.info("✅ Ensemble inference completed")


# ============================================================
# POSTPROCESSING
# ============================================================

def load_prediction(pred_path: Path) -> np.ndarray:
    nifti = nib.load(str(pred_path))
    data = nifti.get_fdata()
    return data.astype(np.uint8)


def visualize_prediction(
    original_jpg: Path,
    prediction: np.ndarray
) -> np.ndarray:
    original = cv2.imread(str(original_jpg))
    if original is None:
        raise ValueError("Failed to load original image")

    color_mask = create_color_mask(prediction, COLOR_MAP)
    overlay = overlay_mask_on_image(original, color_mask, alpha=0.6)
    return overlay


# ============================================================
# MAIN PIPELINE
# ============================================================

def predict(image_path: str) -> str:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    dataset_path, images_ts = setup_nnunet_environment()

    case_id = "Inference_0001"
    nifti_path = images_ts / f"{case_id}_0000.nii.gz"

    # Step 1: JPG → NIfTI
    jpg_to_nifti_for_inference(image_path, nifti_path)

    # Step 2: nnU-Net ensemble prediction
    run_ensemble_inference()

    # Step 3: Load prediction
    pred_dir = Path(os.environ["nnUNet_results"]) / DATASET_FOLDER / "predictions"
    pred_file = pred_dir / f"{case_id}.nii.gz"

    if not pred_file.exists():
        raise FileNotFoundError("Prediction file not found")

    prediction = load_prediction(pred_file)

    # Step 4: Visualization
    result = visualize_prediction(image_path, prediction)

    # Step 5: Save output
    output_path = image_path.with_name(image_path.stem + "_segmented.png")
    cv2.imwrite(str(output_path), result)

    logger.info(f"🎉 Inference completed: {output_path}")
    return str(output_path)


# ============================================================
# CLI ENTRY
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python src/inference.py path/to/image.jpg")
        sys.exit(1)

    predict(sys.argv[1])
