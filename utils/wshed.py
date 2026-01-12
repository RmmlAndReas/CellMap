"""Watershed segmentation utilities for CellMap."""
import numpy as np
from skimage import measure
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from utils.logger import TA_logger

logger = TA_logger()


def wshed(img, channel=None, weak_blur=None, strong_blur=None, seeds='mask', min_seed_area=None, is_white_bg=False, fix_scipy_wshed=False, force_dual_pass=False):
    """
    Perform watershed segmentation on an image.
    
    Simplified version for basic use cases.
    """
    if channel is not None and len(img.shape) > 2:
        img = img[..., channel]
    
    if is_white_bg:
        img = 255 - img
    
    if isinstance(seeds, str) and seeds == 'mask':
        # Use the image itself to find markers
        markers = measure.label(img, connectivity=1, background=255)
    elif isinstance(seeds, np.ndarray):
        markers = seeds
    else:
        # Auto-detect local minima/maxima as seeds
        from scipy import ndimage as ndi
        if is_white_bg:
            local_maxi = ndi.maximum_filter(img, size=3) == img
        else:
            local_maxi = ndi.minimum_filter(img, size=3) == img
        markers = measure.label(local_maxi)
    
    # Create distance transform
    mask = (img > 0).astype(bool) if not is_white_bg else (img < 255).astype(bool)
    distance = distance_transform_edt(mask)
    
    # Apply watershed
    result = watershed(-distance, markers, mask=mask)
    
    return result
