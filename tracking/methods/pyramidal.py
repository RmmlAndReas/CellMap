"""Pyramidal registration-based cell matching.

This module implements cell tracking using per-pixel translation matrices
from pyramidal registration.
"""

import numpy as np
from skimage.measure import regionprops
from utils.logger import TA_logger
from measurements.measurements3D.get_point_on_surface_if_centroid_is_bad import point_on_surface
from tracking.utils.tools import assign_random_ID_to_missing_cells
from skimage import measure

logger = TA_logger()

__DEBUG__ = False


def match_cells_by_pyramidal_registration(tracked_cells_t0, labels_t0, labels_t1, translation_matrix, 
                                         mask_t1, assigned_ids=None):
    """
    Match cells between two frames using pyramidal registration translation matrix.
    
    This function uses per-pixel translation vectors from pyramidal registration to match cells
    by translating centroids from frame t0 to find corresponding cells in frame t1.
    
    Args:
        tracked_cells_t0: Tracked cells image from frame t0 (uint32 array with cell IDs)
        labels_t0: Labeled regions for frame t0
        labels_t1: Labeled regions for frame t1
        translation_matrix: Per-pixel translation matrix from pyramidal registration (height x width x 2)
        mask_t1: Mask image for frame t1
        assigned_ids: List of already assigned cell IDs (for random ID assignment)
    
    Returns:
        tracks: Array with matched cell IDs (uint32), same shape as mask_t1
    """
    height = mask_t1.shape[0]
    width = mask_t1.shape[1]
    
    # Get centroids from frame t0
    centroids_t0 = []
    for iii, region in enumerate(regionprops(labels_t0)):
        centroid = point_on_surface(region, labels_t0)
        centroids_t0.append(centroid)
    
    centroids_t0 = np.array(centroids_t0)
    
    if __DEBUG__:
        print(labels_t1.shape)
    
    # Get region properties for t1
    rps_t1_mask = regionprops(labels_t1)
    
    matched_cells = {}
    matched_cells_in_t1 = []
    
    tracks = np.zeros_like(mask_t1, dtype=np.uint32)
    
    # Match cells using translation matrix
    for centroid_t0 in centroids_t0:
        t0 = translation_matrix[int(centroid_t0[0]), int(centroid_t0[1]), 0]
        t1 = translation_matrix[int(centroid_t0[0]), int(centroid_t0[1]), 1]
        
        try:
            cell_id_in_t0 = tracked_cells_t0[int(centroid_t0[0]), int(centroid_t0[1])]
            translation_corrected_y = int(centroid_t0[0] + t0)
            translation_corrected_x = int(centroid_t0[1] + t1)
            
            if not (translation_corrected_y < height and translation_corrected_y >= 0 and 
                   translation_corrected_x < width and translation_corrected_x >= 0):
                continue
            
            possible_matching_cell = labels_t1[translation_corrected_y, translation_corrected_x]
            
            if possible_matching_cell > 0:
                if possible_matching_cell in matched_cells_in_t1:
                    continue
                if cell_id_in_t0 == 0xFFFFFF:
                    continue
                tracks[labels_t1 == possible_matching_cell] = cell_id_in_t0
                matched_cells_in_t1.append(possible_matching_cell)
                matched_cells[labels_t0[int(centroid_t0[0]), int(centroid_t0[1])]] = possible_matching_cell
        except:
            import traceback
            traceback.print_exc()
    
    # Find matched cell IDs
    matched_cells_ids = []
    for iii, region in enumerate(rps_t1_mask):
        color = tracks[region.coords[0][0], region.coords[0][1]]
        if color == 0:
            pass
        else:
            matched_cells_ids.append(color)
    
    # Find unmatched cells in t0
    labels_tracking_t0 = measure.label(tracked_cells_t0, connectivity=1, background=0xFFFFFF)
    rps_t0 = regionprops(labels_tracking_t0)
    unmatched_cells_in_t0 = []
    for iii, region in enumerate(rps_t0):
        color = tracked_cells_t0[region.coords[0][0], region.coords[0][1]]
        if color not in matched_cells_ids:
            unmatched_cells_in_t0.append(iii)
    
    # Assign random IDs to unmatched cells
    tracks = assign_random_ID_to_missing_cells(tracks, labels_t1, regprps=rps_t1_mask, assigned_ids=assigned_ids)
    tracks[labels_t1 == 0] = 0xFFFFFF
    
    return tracks
