import numpy as np
import cv2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_segmentation_mask(mask, expected_classes=[0, 1, 2, 3]):
    """
    Validate segmentation mask contains only expected class labels
    
    Args:
        mask: 2D numpy array with segmentation labels
        expected_classes: List of valid class IDs
    
    Returns:
        Boolean indicating if mask is valid
    """
    try:
        unique_classes = np.unique(mask)
        
        # Check if all classes are in expected range
        for class_id in unique_classes:
            if class_id not in expected_classes:
                logger.warning(f"Unexpected class ID found: {class_id}")
                return False
        
        # Check mask dimensions
        if len(mask.shape) != 2:
            logger.error(f"Mask should be 2D, got shape: {mask.shape}")
            return False
        
        # Check data type
        if not np.issubdtype(mask.dtype, np.integer):
            logger.warning(f"Mask should be integer type, got: {mask.dtype}")
        
        return True
        
    except Exception as e:
        logger.error(f"Mask validation failed: {e}")
        return False

def check_mask_quality(mask, min_tumor_size=10):
    """
    Check quality of segmentation mask
    
    Args:
        mask: 2D segmentation mask
        min_tumor_size: Minimum number of pixels for valid tumor region
    
    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {
        'valid': True,
        'warnings': [],
        'tumor_regions': {}
    }
    
    try:
        # Check each tumor class
        for class_id in [1, 2, 3]:  # Tumor classes
            class_mask = mask == class_id
            num_pixels = np.sum(class_mask)
            
            quality_metrics['tumor_regions'][class_id] = num_pixels
            
            if num_pixels > 0 and num_pixels < min_tumor_size:
                quality_metrics['warnings'].append(
                    f"Class {class_id} region too small: {num_pixels} pixels"
                )
        
        # Check for fragmented regions
        for class_id in [1, 2, 3]:
            class_mask = (mask == class_id).astype(np.uint8)
            if np.sum(class_mask) > 0:
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 3:  # Too many fragments
                    quality_metrics['warnings'].append(
                        f"Class {class_id} highly fragmented: {len(contours)} regions"
                    )
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"Quality check failed: {e}")
        quality_metrics['valid'] = False
        return quality_metrics

def clean_segmentation_mask(mask, min_region_size=5):
    """
    Clean segmentation mask by removing small isolated regions
    
    Args:
        mask: Input segmentation mask
        min_region_size: Minimum size for regions to keep
    
    Returns:
        Cleaned segmentation mask
    """
    try:
        cleaned_mask = mask.copy()
        
        # Process each tumor class
        for class_id in [1, 2, 3]:
            class_mask = (mask == class_id).astype(np.uint8)
            
            if np.sum(class_mask) == 0:
                continue
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(class_mask)
            
            # Remove small components
            for label_id in range(1, num_labels):
                component_mask = labels == label_id
                if np.sum(component_mask) < min_region_size:
                    cleaned_mask[component_mask] = 0  # Set to background
        
        return cleaned_mask
        
    except Exception as e:
        logger.error(f"Mask cleaning failed: {e}")
        return mask

def save_validation_report(mask, output_path, class_labels):
    """
    Save detailed validation report for segmentation mask
    
    Args:
        mask: Segmentation mask to analyze
        output_path: Path to save report
        class_labels: Dictionary mapping class IDs to names
    """
    try:
        report = []
        report.append("Segmentation Mask Validation Report")
        report.append("=" * 40)
        report.append("")
        
        # Basic statistics
        report.append("Class Distribution:")
        total_pixels = mask.size
        
        for class_id, class_name in class_labels.items():
            class_pixels = np.sum(mask == class_id)
            percentage = (class_pixels / total_pixels) * 100
            report.append(f"  {class_name}: {class_pixels} pixels ({percentage:.2f}%)")
        
        report.append("")
        
        # Quality metrics
        quality = check_mask_quality(mask)
        if quality['warnings']:
            report.append("Quality Warnings:")
            for warning in quality['warnings']:
                report.append(f"  - {warning}")
        else:
            report.append("No quality issues detected.")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Validation report saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save validation report: {e}")