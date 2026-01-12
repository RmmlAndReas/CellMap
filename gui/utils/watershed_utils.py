"""Watershed segmentation operations for paint widget.

This module provides watershed-based segmentation functions for the paint widget,
including guided watershed and manual reseeding operations.
"""

import numpy as np
import logging
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries, watershed
from scipy.ndimage import distance_transform_edt

# Use standard logging to match other modules
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(filename)s - %(funcName)s - line %(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def apply_watershed(lab_mask, min_seed_area=0, guide_mask=None):
    """
    Apply scikit-image watershed to labeled mask.
    
    Args:
        lab_mask: Labeled mask (0 = background/boundaries, >0 = regions)
        min_seed_area: Minimum area for regions (regions smaller than this will be removed)
        guide_mask: Optional binary mask to guide watershed (255 = guide path, 0 = ignore)
                   Must be same shape as lab_mask
            
    Returns:
        Labeled mask after watershed and filtering
    """
    if min_seed_area <= 0:
        # No filtering needed, just apply watershed
        # Create distance transform: boundaries are 0, interiors have high values
        # Invert the mask: boundaries (0) become high, regions become low
        mask = (lab_mask == 0).astype(np.uint8)
        distance = distance_transform_edt(mask)
        
        # If guide_mask is provided, modify distance to favor the guide path
        if guide_mask is not None and guide_mask.shape == distance.shape:
            # Lower distance values along guide path to encourage watershed to follow it
            distance[guide_mask > 0] = np.maximum(distance[guide_mask > 0] - 2, 1)
        
        # Apply watershed using the labeled regions as markers
        # Include guide path in the mask so watershed can expand into it
        if guide_mask is not None and guide_mask.shape == lab_mask.shape:
            watershed_mask = (lab_mask > 0) | (guide_mask > 0)
            logger.info(f'apply_watershed: Using guide mask, expanded mask from {np.count_nonzero(lab_mask > 0)} to {np.count_nonzero(watershed_mask)} pixels')
        else:
            watershed_mask = (lab_mask > 0)
        result = watershed(-distance, lab_mask, mask=watershed_mask)
        return result
    else:
        # Apply watershed first
        mask = (lab_mask == 0).astype(np.uint8)
        distance = distance_transform_edt(mask)
        
        # If guide_mask is provided, modify distance to favor the guide path
        if guide_mask is not None and guide_mask.shape == distance.shape:
            distance[guide_mask > 0] = np.maximum(distance[guide_mask > 0] - 2, 1)
        
        # Include guide path in the mask so watershed can expand into it
        if guide_mask is not None and guide_mask.shape == lab_mask.shape:
            watershed_mask = (lab_mask > 0) | (guide_mask > 0)
            logger.info(f'apply_watershed: Using guide mask with filtering, expanded mask from {np.count_nonzero(lab_mask > 0)} to {np.count_nonzero(watershed_mask)} pixels')
        else:
            watershed_mask = (lab_mask > 0)
        result = watershed(-distance, lab_mask, mask=watershed_mask)
        
        # Filter small regions
        rps = regionprops(result)
        filtered_result = np.zeros_like(result)
        current_label = 1
        for rp in rps:
            if rp.area >= min_seed_area:
                filtered_result[result == rp.label] = current_label
                current_label += 1
        
        return filtered_result


def manually_reseeded_wshed(paint_widget):
    """
    Apply manually reseeded watershed using user drawing as seeds.
    
    This is the local seeded watershed à la TA.
    
    Args:
        paint_widget: The paint widget instance (must have get_user_drawing, get_mask, 
                     get_raw_image, channel, set_mask methods)
    """
    if paint_widget.get_raw_image() is None:
        return

    try:
        if paint_widget.channel is None and paint_widget.get_raw_image().has_c():
            logger.error('Please select a channel fisrt')
            return
    except:
        print(paint_widget.get_raw_image().shape, paint_widget.channel)
        if paint_widget.channel is None and len(paint_widget.get_raw_image().shape) >= 3 and paint_widget.get_raw_image().shape[-1] > 1:
            logger.error('Please select a channel fisrt')
            return

    usr_drawing = paint_widget.get_user_drawing()
    labs = label(usr_drawing, connectivity=2, background=0)

    rps_user_drawing = regionprops(labs)
    if len(rps_user_drawing) == 0:
        return

    drawn_mask = paint_widget.get_mask()

    drawn_mask[labs != 0] = 0
    lab_cells_user_drawing = label(drawn_mask, connectivity=1, background=255)
    rps_cells_user_drawing = regionprops(lab_cells_user_drawing)

    min_y = 100000000
    min_x = 100000000
    max_y = 0
    max_x = 0
    for rps_user in rps_user_drawing:
        label_id = lab_cells_user_drawing[rps_user.coords[0][0], rps_user.coords[0][1]]
        bbox = rps_cells_user_drawing[label_id - 1].bbox
        min_y = min(bbox[0], min_y)
        min_x = min(bbox[1], min_x)
        max_y = max(bbox[2], max_y)
        max_x = max(bbox[3], max_x)

    # here I need get the channel in fact
    img = paint_widget.get_raw_image()[min_y:max_y + 1, min_x:max_x + 1]
    if paint_widget.channel is not None:
        img = img[..., paint_widget.channel]
    # Use scikit-image watershed with image and seeds
    seeds = labs[min_y:max_y + 1, min_x:max_x + 1]
    # Create mask from seeds (non-zero regions)
    mask = (seeds > 0).astype(bool)
    # Apply watershed using image as distance/input and seeds as markers
    minished = watershed(-img, seeds, mask=mask)
    drawn_mask[min_y:max_y + 1, min_x:max_x + 1][minished != 0] = minished[minished != 0]
    paint_widget.set_mask(drawn_mask)

