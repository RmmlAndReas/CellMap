"""Core tracking orchestrator.

This module provides the main tracking functions that orchestrate different
tracking methods and apply error correction.
"""

from utils.image_utils import RGB_to_int24, int24_to_RGB
from timeit import default_timer as timer
import traceback
import os
import numpy as np
from skimage.measure import label, regionprops
from utils.image_io import Img
from utils.logger import TA_logger

logger = TA_logger()
from colors.colorgen import get_forbidden_colors_int24
from skimage import measure

# Import tracking methods
from tracking.methods.pyramidal import match_cells_by_pyramidal_registration

# Import utilities
from tracking.utils.registration import get_pyramidal_registration, pre_register_images, apply_translation
from tracking.utils.tools import (
    assign_random_ID_to_missing_cells,
    first_image_tracking,
    get_list_of_files,
    get_input_files_and_output_folders,
    get_TA_file,
    get_mask_file,
    smart_name_parser,
)

# Import correction functions
from tracking.correction.tracking_error_detector_and_fixer import (
    find_vertices,
    associate_cell_to_its_neighbors2,
    associate_cells_to_neighbors_ID_in_dict,
    compute_neighbor_score,
    optimize_score,
    get_cells_in_image_n_fisrt_pixel,
    map_track_id_to_label,
    apply_color_to_labels,
)

from utils.early_stopper import early_stop
from database.sqlite_db import save_track_confidence, get_master_db_path

# Tracking method registry
TRACKING_METHODS = {
    'pyramidal': match_cells_by_pyramidal_registration,
}

# parameters
__DEBUG__ = False


def track_cells_dynamic_tissue(path, channel=None, PYRAMIDAL_DEPTH=3, THRESHOLD_TRANSLATION=20, MAX_ITER=15, progress_callback=None, tracking_method='pyramidal'):
    """
    Track cells across a sequence of images using the specified tracking method.
    
    Args:
        path: Path to directory or list file containing images to track
        channel: Channel index for registration (optional)
        PYRAMIDAL_DEPTH: Depth of pyramidal registration (default: 3)
        THRESHOLD_TRANSLATION: Maximum translation (pixels) to accept during pyramidal registration (default: 20)
        MAX_ITER: Maximum number of optimization iterations (default: 15)
        progress_callback: Optional callback for progress updates
        tracking_method: Tracking method to use ('pyramidal', default: 'pyramidal')
    
    Returns:
        None (saves tracked results to files)
    """
    start_all = timer()

    seed = 1  # always start tracking with same seed to have roughly the same color
    assigned_ids = get_forbidden_colors_int24()

    # Get list of images to analyze
    images_to_analyze = get_list_of_files(path)

    # Handle single image case
    if len(images_to_analyze) == 1:
        filename0_without_ext = smart_name_parser(images_to_analyze[0], 'full_no_ext')
        mask_file_path = get_mask_file(filename0_without_ext)
        if not os.path.exists(mask_file_path):
            logger.error(f"Could not find mask file (handCorrection.tif): {mask_file_path}")
            return
        mask_t0 = Img(mask_file_path)
        if mask_t0 is None:
            logger.error(f"Could not load mask file: {mask_file_path}")
            return
        if mask_t0.has_c():
            mask_t0 = mask_t0[..., 0]
        labels_t0 = measure.label(mask_t0, connectivity=1, background=255)  # FOUR_CONNECTED
        tracked_cells_t0 = first_image_tracking(mask_t0, labels_t0, assigned_ids=assigned_ids, seed=seed)
        Img(int24_to_RGB(tracked_cells_t0), dimensions='hwc').save(
            get_TA_file(filename0_without_ext, 'tracked_cells_resized.tif'),
            mode='raw')
        return

    # Validate tracking method
    if tracking_method not in TRACKING_METHODS:
        raise ValueError(f"Unknown tracking method: {tracking_method}. Must be one of: {list(TRACKING_METHODS.keys())}")

    # Get the tracking function
    match_function = TRACKING_METHODS[tracking_method]

    # Process each frame pair
    for l in range(len(images_to_analyze) - 1):
        try:
            if early_stop.stop == True:
                return
            if progress_callback is not None:
                progress_callback.emit(int((l * 100) / len(images_to_analyze)))
            else:
                print(str((l * 100) / len(images_to_analyze)) + '%')
        except:
            pass

        start_loop = timer()

        file_path_0, file_path_1, filename0_without_ext, filename1_without_ext = get_input_files_and_output_folders(
            images_to_analyze, l)

        orig_t0 = Img(file_path_0)
        orig_t1 = Img(file_path_1)
        if channel is not None:
            # we reensure image has channel otherwise skip
            if len(orig_t0.shape) > 2:
                orig_t0 = orig_t0[..., channel]
            if len(orig_t1.shape) > 2:
                orig_t1 = orig_t1[..., channel]
        
        # Pyramidal registration to compute translation matrix
        print('intermediate time before pyramidal registration', timer() - start_loop)
        
        # Try to load manual landmarks for this frame pair
        landmarks_t0 = None
        landmarks_t1 = None
        try:
            from database.sqlite_db import get_landmarks_for_frame_pair
            master_db_path = get_master_db_path([file_path_0])
            if master_db_path:
                landmarks_t0, landmarks_t1 = get_landmarks_for_frame_pair(
                    master_db_path, filename0_without_ext, filename1_without_ext
                )
                if landmarks_t0 and landmarks_t1:
                    logger.info(f'Using {len(landmarks_t0)} manual landmarks for registration between {filename0_without_ext} and {filename1_without_ext}')
        except Exception as e:
            logger.warning(f'Could not load landmarks for frame pair: {e}')
            landmarks_t0 = None
            landmarks_t1 = None
        
        translation_matrix = get_pyramidal_registration(
            orig_t0, orig_t1, 
            depth=PYRAMIDAL_DEPTH, 
            threshold_translation=THRESHOLD_TRANSLATION,
            landmarks_t0=landmarks_t0,
            landmarks_t1=landmarks_t1
        )
        print('intermediate time after pyramidal registration', timer() - start_loop)
        
        # Save translation matrix as TIF file
        translation_matrix_path = get_TA_file(filename0_without_ext, 'translation_matrix.tif')
        translation_matrix_img = Img(translation_matrix.astype(np.float32), dimensions='hwc')
        translation_matrix_img.save(translation_matrix_path, mode='IJ')
        logger.info(f"Saved translation matrix to: {translation_matrix_path}")

        # Load masks
        mask_file_path = get_mask_file(filename0_without_ext)
        if not os.path.exists(mask_file_path):
            logger.error(f"Could not find mask file (handCorrection.tif): {mask_file_path}")
            continue
        mask_t0 = Img(mask_file_path)
        if mask_t0 is None:
            logger.error(f"Could not load mask file: {mask_file_path}")
            continue
        if mask_t0.has_c():
            mask_t0 = mask_t0[..., 0]
        labels_t0 = measure.label(mask_t0, connectivity=1, background=255)  # FOUR_CONNECTED

        if __DEBUG__:
            print(mask_t0.shape, mask_file_path)

        mask_t1_path = get_mask_file(filename1_without_ext)
        if not os.path.exists(mask_t1_path):
            logger.error(f"Could not find mask file (handCorrection.tif): {mask_t1_path}")
            continue
        mask_t1 = Img(mask_t1_path)
        if mask_t1.has_c():
            mask_t1 = mask_t1[..., 0]
        
        # Initialize or load tracked cells for frame t0
        if l == 0:
            tracked_cells_t0 = first_image_tracking(mask_t0, labels_t0, assigned_ids=assigned_ids, seed=seed)
            Img(int24_to_RGB(tracked_cells_t0), dimensions='hwc').save(
                get_TA_file(filename0_without_ext, 'tracked_cells_resized.tif'),
                mode='raw')
        else:
            tracked_cells_t0 = RGB_to_int24(Img(get_TA_file(filename0_without_ext, 'tracked_cells_resized.tif')))

        if __DEBUG__:
            print(mask_t0.shape)
            print(mask_t1.shape)

        # Label the mask for frame t1
        labels_t1 = measure.label(mask_t1, connectivity=1, background=255)  # FOUR_CONNECTED
        
        if __DEBUG__:
            print(labels_t1.shape)
        
        # Get region properties for t1 (needed for assign_random_ID_to_missing_cells)
        rps_t1_mask = regionprops(labels_t1)
        
        # Apply tracking method
        print(f'intermediate time before {tracking_method} matching', timer() - start_loop)
        
        try:
            tracks = match_function(
                tracked_cells_t0=tracked_cells_t0,
                labels_t0=labels_t0,
                labels_t1=labels_t1,
                translation_matrix=translation_matrix,
                mask_t1=mask_t1,
                assigned_ids=assigned_ids
            )
            
            tracked_cells_t1 = tracks
            print(f'intermediate time after {tracking_method} matching', timer() - start_loop)
        except Exception as e:
            logger.error(f"Tracking method {tracking_method} failed: {e}. Skipping tracking for this frame pair.")
            traceback.print_exc()
            continue

        print('intermediate time before swap correction0', timer() - start_loop)

        # Apply swapping correction using neighbor-based optimization
        track_t_cur = tracks
        track_t_minus_1 = tracked_cells_t0

        # Get vertices for neighbor analysis
        vertices_t_minus_1 = np.stack(np.where(track_t_minus_1 == 0xFFFFFF), axis=1)
        vertices_t_cur = np.stack(np.where(track_t_cur == 0xFFFFFF), axis=1)

        # Build neighbor associations
        cells_and_their_neighbors_minus_1 = np.unique(
            np.asarray(associate_cell_to_its_neighbors2(vertices_t_minus_1, track_t_minus_1)), axis=0)
        cells_and_their_neighbors_minus_1 = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_minus_1)
        cells_in_t_minus_1 = np.unique(track_t_minus_1)

        initial_score = -1
        corrected_track = track_t_cur
        final_confidence_scores = {}

        # Optimization loop
        for lll in range(MAX_ITER):
            cells_and_their_neighbors_cur = np.unique(
                np.asarray(associate_cell_to_its_neighbors2(vertices_t_cur, track_t_cur)), axis=0)
            cells_and_their_neighbors_cur = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_cur)

            cells_present_in_t_cur_but_absent_in_t_minus_1 = []
            cells_present_in_t_cur_but_absent_in_t_minus_1_score = {}

            score = compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
                                           cells_present_in_t_cur_but_absent_in_t_minus_1,
                                           cells_present_in_t_cur_but_absent_in_t_minus_1_score)

            if initial_score < 0:
                initial_score = score

            lab_t_cur = label(track_t_cur, connectivity=1, background=0xFFFFFF)
            cells_in_t_cur, first_pixels_t_cur = get_cells_in_image_n_fisrt_pixel(track_t_cur)
            map_tracks_n_label_t_cur = map_track_id_to_label(first_pixels_t_cur, track_t_cur, lab_t_cur)

            FIX_SWAPS = True
            threshold = 0.8
            if FIX_SWAPS:
                cells_and_their_neighbors_cur, map_tracks_n_label_t_cur, last_score_reached = optimize_score(
                    cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
                    cells_present_in_t_cur_but_absent_in_t_minus_1_score, map_tracks_n_label_t_cur, threshold=threshold)

                if last_score_reached == initial_score:
                    print('early stop', lll)
                    break

                if last_score_reached > initial_score:
                    print('score improved from', initial_score, 'to', last_score_reached)
                    initial_score = last_score_reached
                else:
                    print('score did not improve', initial_score)

                compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
                                       cells_present_in_t_cur_but_absent_in_t_minus_1,
                                       cells_present_in_t_cur_but_absent_in_t_minus_1_score)
                
                final_confidence_scores = cells_present_in_t_cur_but_absent_in_t_minus_1_score.copy()

                corrected_track = apply_color_to_labels(lab_t_cur, map_tracks_n_label_t_cur)
                corrected_track = assign_random_ID_to_missing_cells(corrected_track, labels_t1, regprps=rps_t1_mask,
                                                                    assigned_ids=assigned_ids)
                track_t_cur = corrected_track

            print('end loop', timer() - start_loop)

        # Save tracking confidence scores to master database
        if l > 0 and final_confidence_scores:
            try:
                unique_tracks = np.unique(track_t_cur)
                unique_tracks = unique_tracks[unique_tracks != 0xFFFFFF]
                
                final_confidence = {}
                for track_id in unique_tracks:
                    if track_id in final_confidence_scores:
                        final_confidence[track_id] = final_confidence_scores[track_id]
                    else:
                        final_confidence[track_id] = 0.0
                
                if final_confidence:
                    master_db_path = get_master_db_path([images_to_analyze[l+1]])
                    if master_db_path:
                        save_result = save_track_confidence(final_confidence, l+1, master_db_path)
            except Exception as e:
                logger.warning(f'Error saving track confidence for frame {l+1}: {e}')
                traceback.print_exc()

        # Save tracked result
        print(get_TA_file(filename1_without_ext, 'tracked_cells_resized.tif'))
        Img(int24_to_RGB(corrected_track)).save(get_TA_file(filename1_without_ext, 'tracked_cells_resized.tif'),
                                                mode='raw')

    print('total time', timer() - start_all)


def _pre_reg_static(name_t0, name_t1, channel_of_interest):
    """
    Perform pre-registration between two images for static tracking.
    
    Args:
        name_t0: Filename of the first image
        name_t1: Filename of the second image
        channel_of_interest: Channel index (optional)
    
    Returns:
        tuple: Translation dimensions (trans_dim_0, trans_dim_1)
    """
    I0 = Img(name_t0)
    if channel_of_interest is not None and len(I0.shape) > 2:
        I0 = I0[..., channel_of_interest]
    I1 = Img(name_t1)
    if channel_of_interest is not None and len(I1.shape) > 2:
        I1 = I1[..., channel_of_interest]
    trans_dim_0, trans_dim_1 = pre_register_images(orig_t0=I0, orig_t1=I1)
    return trans_dim_0, trans_dim_1


def _apply_mermaid_warp_if_available(tracked_cells_t0, filename0_without_ext, name_t0, name_t1, 
                                     channel_of_interest, pre_register):
    """
    Apply Mermaid warping to tracked cells if map is available, otherwise apply pre-registration.
    
    Args:
        tracked_cells_t0: Tracked cells image from frame t0
        filename0_without_ext: Filename without extension for frame t0
        name_t0: Full filename for frame t0 (for pre-registration)
        name_t1: Full filename for frame t1 (for pre-registration)
        channel_of_interest: Channel index for pre-registration
        pre_register: Whether to apply pre-registration if Mermaid is not available
    
    Returns:
        warped_tracked_cells: Warped tracked cells image
    """
    warped_tracked_cells = tracked_cells_t0.copy()
    
    try:
        from personal.mermaid.deep_warping_uing_mermaid_minimal import warp_image_directly_using_phi
        
        mermaid_map_path = os.path.join(filename0_without_ext, 'mermaid_map.tif')
        if os.path.exists(mermaid_map_path):
            # Warp the cell tracks using Mermaid if the map is available
            warp_map = Img(mermaid_map_path)
            warped_tracked_cells = warp_image_directly_using_phi(
                warped_tracked_cells, 
                warp_map, 
                pre_registration=name_t0 if pre_register else None
            ).astype(np.uint32)
        else:
            if pre_register:
                # Perform pre-registration if specified
                trans_dim_0, trans_dim_1 = _pre_reg_static(name_t0, name_t1, channel_of_interest)
                warped_tracked_cells = apply_translation(warped_tracked_cells, -trans_dim_0, -trans_dim_1)
    except Exception as e:
        logger.warning(f"Mermaid registration failed: {e}. Continuing without.")
        if pre_register:
            # Perform pre-registration if specified
            trans_dim_0, trans_dim_1 = _pre_reg_static(name_t0, name_t1, channel_of_interest)
            warped_tracked_cells = apply_translation(warped_tracked_cells, -trans_dim_0, -trans_dim_1)
    
    return warped_tracked_cells


def match_by_max_overlap(name_t1, name_t0, channel_of_interest=None, assigned_IDs=[], 
                         recursive_assignment=True, warp_using_mermaid_if_map_is_available=True, 
                         pre_register=True):
    """
    Match cells between two timepoints based on maximizing overlap (static tissue tracking).
    
    This is a file-based wrapper around the static overlap matching method. It handles
    file loading, Mermaid warping, and saving results.
    
    Args:
        name_t1: Filename of the first timepoint
        name_t0: Filename of the second timepoint
        channel_of_interest: Channel index for registration (optional)
        assigned_IDs: List of already assigned cell IDs
        recursive_assignment: Whether to perform recursive assignment for error correction
        warp_using_mermaid_if_map_is_available: Whether to warp using Mermaid if map is available
        pre_register: Whether to perform pre-registration
    
    Returns:
        None (saves tracked results to files)
    """
    # Get the filename of the first timepoint without the extension
    filename1_without_ext = os.path.splitext(name_t1)[0]
    
    # Load the hand-corrected mask for the first timepoint
    mask_file_path = get_mask_file(filename1_without_ext)
    if not os.path.exists(mask_file_path):
        # Fallback to direct path if get_mask_file didn't find it
        mask_file_path = os.path.join(filename1_without_ext, 'handCorrection.tif')
    mask_t1 = Img(mask_file_path)
    
    # Load the tracked cells for the second timepoint
    tracked_cells_t0 = RGB_to_int24(Img(os.path.join(os.path.splitext(name_t0)[0], 'tracked_cells_resized.tif')))
    
    # Apply warping (Mermaid or pre-registration)
    filename0_without_ext = os.path.splitext(name_t0)[0]
    if warp_using_mermaid_if_map_is_available:
        tracked_cells_t0_warped = _apply_mermaid_warp_if_available(
            tracked_cells_t0, filename0_without_ext, name_t0, name_t1, 
            channel_of_interest, pre_register
        )
    else:
        if pre_register:
            trans_dim_0, trans_dim_1 = _pre_reg_static(name_t0, name_t1, channel_of_interest)
            tracked_cells_t0_warped = apply_translation(tracked_cells_t0, -trans_dim_0, -trans_dim_1)
        else:
            tracked_cells_t0_warped = tracked_cells_t0
    
    # Convert the mask to label image
    if len(mask_t1.shape) == 3:
        mask_t1 = mask_t1[..., 0]
    
    labels_t1 = measure.label(mask_t1, connectivity=1, background=255)
    rps_label_t1 = regionprops(labels_t1)
    
    # Load original images for pyramidal registration
    orig_t0 = Img(name_t0)
    orig_t1 = Img(name_t1)
    if channel_of_interest is not None:
        if len(orig_t0.shape) > 2:
            orig_t0 = orig_t0[..., channel_of_interest]
        if len(orig_t1.shape) > 2:
            orig_t1 = orig_t1[..., channel_of_interest]
    
    # Get labels for t0 from tracked cells
    labels_t0 = measure.label(tracked_cells_t0_warped, connectivity=1, background=0xFFFFFF)
    
    # Compute pyramidal registration
    translation_matrix = get_pyramidal_registration(
        orig_t0, orig_t1, 
        depth=3, 
        threshold_translation=20
    )
    
    # Apply pyramidal matching
    track_t1 = match_cells_by_pyramidal_registration(
        tracked_cells_t0=tracked_cells_t0_warped,
        labels_t0=labels_t0,
        labels_t1=labels_t1,
        translation_matrix=translation_matrix,
        mask_t1=mask_t1,
        assigned_ids=assigned_IDs
    )
    
    if recursive_assignment:
        # Perform recursive assignment to further identify cells
        run_swapping_correction_recursively_to_further_identify_cells(
            track_t1, tracked_cells_t0_warped, labels_t1, rps_label_t1, 
            assigned_IDs, filename1_without_ext, MAX_ITER=15
        )
    else:
        # Save the tracked cells for the first timepoint
        Img(int24_to_RGB(track_t1)).save(
            get_TA_file(filename1_without_ext, 'tracked_cells_resized.tif'), 
            mode='raw'
        )


def run_swapping_correction_recursively_to_further_identify_cells(tracks, tracked_cells_t0, labels_t1, rps_t1_mask,
                                                                  assigned_ids, filename1_without_ext, MAX_ITER=15):
    """
    Apply recursive swapping correction to improve tracking accuracy.
    
    This function uses neighbor-based optimization to detect and fix cell ID swaps.
    
    Args:
        tracks: Current tracked cells image
        tracked_cells_t0: Tracked cells from previous frame
        labels_t1: Labeled regions for frame t1
        rps_t1_mask: Region properties for frame t1
        assigned_ids: List of assigned cell IDs
        filename1_without_ext: Filename without extension for saving
        MAX_ITER: Maximum number of optimization iterations
    """
    start_all = timer()
    
    # just for mapping
    track_t_cur = tracks
    track_t_minus_1 = tracked_cells_t0
    
    # Get vertices for neighbor analysis
    vertices_t_minus_1 = np.stack(np.where(track_t_minus_1 == 0xFFFFFF), axis=1)
    vertices_t_cur = np.stack(np.where(track_t_cur == 0xFFFFFF), axis=1)
    
    # Build neighbor associations
    cells_and_their_neighbors_minus_1 = np.unique(
        np.asarray(associate_cell_to_its_neighbors2(vertices_t_minus_1, track_t_minus_1)), axis=0)
    cells_and_their_neighbors_minus_1 = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_minus_1)
    
    initial_score = -1
    
    # Optimization loop
    for lll in range(MAX_ITER):
        start_loop = timer()
        
        cells_and_their_neighbors_cur = np.unique(
            np.asarray(associate_cell_to_its_neighbors2(vertices_t_cur, track_t_cur)), axis=0)
        cells_and_their_neighbors_cur = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_cur)
        
        cells_present_in_t_cur_but_absent_in_t_minus_1 = []
        cells_present_in_t_cur_but_absent_in_t_minus_1_score = {}
        
        score = compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
                                       cells_present_in_t_cur_but_absent_in_t_minus_1,
                                       cells_present_in_t_cur_but_absent_in_t_minus_1_score)
        
        if initial_score < 0:
            initial_score = score
        
        lab_t_cur = measure.label(track_t_cur, connectivity=1, background=0xFFFFFF)
        cells_in_t_cur, first_pixels_t_cur = get_cells_in_image_n_fisrt_pixel(track_t_cur)
        map_tracks_n_label_t_cur = map_track_id_to_label(first_pixels_t_cur, track_t_cur, lab_t_cur)
        
        FIX_SWAPS = True
        threshold = 0.8
        
        if FIX_SWAPS:
            cells_and_their_neighbors_cur, map_tracks_n_label_t_cur, last_score_reached = optimize_score(
                cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
                cells_present_in_t_cur_but_absent_in_t_minus_1_score, map_tracks_n_label_t_cur, threshold=threshold)
            
            if last_score_reached == initial_score:
                print('early stop', lll)
                break
            
            if last_score_reached > initial_score:
                print('score improved from', initial_score, 'to', last_score_reached)
                initial_score = last_score_reached
            else:
                print('score did not improve', initial_score)
            
            compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
                                   cells_present_in_t_cur_but_absent_in_t_minus_1,
                                   cells_present_in_t_cur_but_absent_in_t_minus_1_score)
            
            corrected_track = apply_color_to_labels(lab_t_cur, map_tracks_n_label_t_cur)
            corrected_track = assign_random_ID_to_missing_cells(corrected_track, labels_t1, regprps=rps_t1_mask,
                                                                assigned_ids=assigned_ids)
            track_t_cur = corrected_track
        
        print('end loop', timer() - start_loop)
    
    # Make sure to fix missing not found cells
    track_t_cur = assign_random_ID_to_missing_cells(track_t_cur, labels_t1, regprps=rps_t1_mask,
                                                 assigned_ids=assigned_ids)
    
    # Save the result
    print(get_TA_file(filename1_without_ext, 'tracked_cells_resized.tif'))
    Img(int24_to_RGB(track_t_cur)).save(get_TA_file(filename1_without_ext, 'tracked_cells_resized.tif'), mode='raw')
    
    print('total time all recursions', timer() - start_all)


def track_first_frame_static(original_t0, assigned_ids, seed):
    """
    Track the first frame for static tissue tracking.
    
    Args:
        original_t0: Filename of the first image
        assigned_ids: List of assigned cell IDs
        seed: Random seed for consistent color assignment
    """
    filename0_without_ext = os.path.splitext(original_t0)[0]
    mask_file_path = get_mask_file(filename0_without_ext)
    if not os.path.exists(mask_file_path):
        # Fallback to direct path if get_mask_file didn't find it
        mask_file_path = get_TA_file(filename0_without_ext, 'handCorrection.tif')
    mask_t0 = Img(mask_file_path)
    
    if mask_t0.has_c():
        mask_t0 = mask_t0[..., 0]
    labels_t0 = measure.label(mask_t0, connectivity=1, background=255)  # FOUR_CONNECTED
    tracked_cells_t0 = first_image_tracking(mask_t0, labels_t0, assigned_ids=assigned_ids, seed=seed)
    Img(int24_to_RGB(tracked_cells_t0), dimensions='hwc').save(
        get_TA_file(filename0_without_ext, 'tracked_cells_resized.tif'), mode='raw')


def track_cells_static_tissue(lst, channel_of_interest=None, recursive_assignment=True,
                              warp_using_mermaid_if_map_is_available=True, pre_register=True,
                              progress_callback=None):
    """
    Track cells across a sequence of images for static tissue.
    
    This function performs tracking for static tissues where cells have minimal
    movement between frames. Supports Mermaid warping and pre-registration.
    
    Args:
        lst: List of image filenames to track
        channel_of_interest: Channel index for registration (optional)
        recursive_assignment: Whether to use recursive assignment for error correction
        warp_using_mermaid_if_map_is_available: Whether to warp using Mermaid if map is available
        pre_register: Whether to perform pre-registration
        progress_callback: Optional callback for progress updates
    
    Returns:
        None (saves tracked results to files)
    """
    start_all = timer()
    seed = 1  # always start tracking with the same seed to have roughly the same color
    assigned_ids = get_forbidden_colors_int24()
    
    try:
        zipped_list = zip(lst, lst[1:])
    except:
        # Assume list contains only one image --> return just the first image tracking
        track_first_frame_static(lst[0], assigned_ids, seed)
        return
    
    for iii, (original_t0, original_t1) in enumerate(zipped_list):
        try:
            if early_stop.stop:
                return
            if progress_callback is not None:
                progress_callback.emit(int((iii * 100) / len(zipped_list)))
            else:
                print(str((iii * 100) / len(zipped_list)) + '%')
        except:
            pass
        
        if iii == 0:
            # Need to create the track for the first image then recursively go on
            track_first_frame_static(original_t0, assigned_ids, seed)
        
        match_by_max_overlap(
            original_t1, original_t0, 
            channel_of_interest=channel_of_interest,
            recursive_assignment=recursive_assignment,
            warp_using_mermaid_if_map_is_available=warp_using_mermaid_if_map_is_available,
            pre_register=pre_register
        )
    
    print('total time', timer() - start_all)
