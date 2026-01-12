# import numpy as np
# import cv2
# import logging

# logger = logging.getLogger(__name__)

# def create_color_mask(prediction, color_map):
#     """
#     Convert segmentation prediction to color-coded mask
    
#     Args:
#         prediction: 2D numpy array with class labels
#         color_map: Dictionary mapping class labels to RGB colors
    
#     Returns:
#         3D numpy array (H, W, 3) with color-coded segmentation
#     """
#     height, width = prediction.shape
#     color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
#     for class_id, color in color_map.items():
#         mask = prediction == class_id
#         color_mask[mask] = color
    
#     return color_mask

# def resize_image(image, target_size=(512, 512)):
#     """
#     Resize image to target size while maintaining aspect ratio
    
#     Args:
#         image: Input image (numpy array)
#         target_size: Tuple of (width, height)
    
#     Returns:
#         Resized image
#     """
#     return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

# def normalize_image(image):
#     """
#     Normalize image to 0-1 range
    
#     Args:
#         image: Input image (numpy array)
    
#     Returns:
#         Normalized image
#     """
#     image = image.astype(np.float32)
#     image = (image - image.min()) / (image.max() - image.min() + 1e-8)
#     return image

# def preprocess_ct_image(image):
#     """
#     Preprocess CT image for nnU-Net
    
#     Args:
#         image: Input CT image (numpy array)
    
#     Returns:
#         Preprocessed image
#     """
#     # Convert to grayscale if needed
#     if len(image.shape) == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Normalize
#     image = normalize_image(image)
    
#     # Resize to standard size
#     image = resize_image(image)
    
#     return image

# def validate_image(image_path):
#     """
#     Validate that the image file is readable and has correct format
    
#     Args:
#         image_path: Path to image file
    
#     Returns:
#         Boolean indicating if image is valid
#     """
#     try:
#         image = cv2.imread(image_path)
#         if image is None:
#             return False
        
#         # Check dimensions
#         if len(image.shape) not in [2, 3]:
#             return False
        
#         # Check size
#         height, width = image.shape[:2]
#         if height < 64 or width < 64:
#             logger.warning(f"Image too small: {width}x{height}")
#             return False
        
#         return True
        
#     except Exception as e:
#         logger.error(f"Image validation failed: {e}")
#         return False

# def calculate_tumor_statistics(prediction, class_labels):
#     """
#     Calculate statistics about detected tumors
    
#     Args:
#         prediction: 2D segmentation mask
#         class_labels: Dictionary mapping class IDs to names
    
#     Returns:
#         Dictionary with tumor statistics
#     """
#     stats = {}
#     total_pixels = prediction.size
    
#     for class_id, class_name in class_labels.items():
#         if class_id == 0:  # Skip background
#             continue
            
#         class_pixels = np.sum(prediction == class_id)
#         percentage = (class_pixels / total_pixels) * 100
        
#         stats[class_name] = {
#             'pixels': int(class_pixels),
#             'percentage': round(percentage, 2)
#         }
    
#     return stats


import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# ============================================================
# ---------------- TRAINING-SAFE UTILITIES ------------------
# ============================================================

def preprocess_ct_image_for_training(image: np.ndarray) -> np.ndarray:
    """
    Minimal preprocessing for nnU-Net training.
    IMPORTANT: nnU-Net expects near-raw data.

    - NO resizing
    - NO normalization
    - NO histogram equalization

    Args:
        image: 2D numpy array (grayscale CT slice)

    Returns:
        Float32 image suitable for nnU-Net
    """
    if image.ndim != 2:
        raise ValueError("CT image must be 2D for nnU-Net 2D")

    return image.astype(np.float32)


def validate_ct_image(image: np.ndarray) -> bool:
    """
    Validate CT image for training/inference.

    Args:
        image: numpy array

    Returns:
        True if valid
    """
    if image is None:
        return False

    if image.ndim not in (2, 3):
        logger.error(f"Invalid image dimensions: {image.shape}")
        return False

    h, w = image.shape[:2]
    if h < 64 or w < 64:
        logger.warning(f"Image too small: {w}x{h}")
        return False

    return True


# ============================================================
# ---------------- INFERENCE / UI UTILITIES -----------------
# ============================================================

def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Normalize image ONLY for visualization (UI).
    Never use this for training.

    Args:
        image: numpy array

    Returns:
        uint8 image [0,255]
    """
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = (image * 255).astype(np.uint8)
    return image


def resize_for_display(image: np.ndarray, size=(512, 512)) -> np.ndarray:
    """
    Resize ONLY for UI display.

    Args:
        image: numpy array
        size: (width, height)

    Returns:
        resized image
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def create_color_mask(
    prediction: np.ndarray,
    color_map: dict
) -> np.ndarray:
    """
    Convert segmentation mask to RGB color image.

    Args:
        prediction: 2D array with class IDs
        color_map: dict {class_id: (R,G,B)}

    Returns:
        RGB mask (H, W, 3)
    """
    h, w = prediction.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        color_mask[prediction == class_id] = color

    return color_mask


def overlay_mask_on_image(
    image: np.ndarray,
    mask_rgb: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay segmentation mask on CT image (for UI).

    Args:
        image: grayscale image
        mask_rgb: RGB segmentation mask
        alpha: blending factor

    Returns:
        overlay image
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
    return overlay


def calculate_tumor_statistics(
    prediction: np.ndarray,
    class_labels: dict
) -> dict:
    """
    Calculate tumor area statistics.

    Args:
        prediction: 2D segmentation mask
        class_labels: {class_id: class_name}

    Returns:
        dict with pixel counts & percentages
    """
    stats = {}
    total_pixels = prediction.size

    for class_id, name in class_labels.items():
        if class_id == 0:
            continue

        count = int(np.sum(prediction == class_id))
        percentage = round((count / total_pixels) * 100, 2)

        stats[name] = {
            "pixels": count,
            "percentage": percentage
        }

    return stats
