"""Cellpose segmentation integration.

This module provides functions to segment images using Cellpose, a deep learning-based
cell segmentation tool. It handles image loading, Cellpose model initialization,
segmentation execution, and output formatting to match the CellMap workflow.
"""

import os
from pathlib import Path
import numpy as np
from skimage.measure import label
from skimage.segmentation import find_boundaries, watershed
from skimage.morphology import remove_small_holes, skeletonize
from utils.image_io import Img
from tracking.tools import smart_name_parser
import logging

# Use TA_logger to ensure logs appear in GUI log box
# Use the master logger which has the GUI handler attached
from utils.logger import TA_logger
import logging
# 
# Function to get logger - ensures we always get the master logger with current handlers
def get_logger():
    """Get the master logger, ensuring it has the GUI handler."""
    logger = TA_logger()  # This returns the master logger
    # Ensure it has the handler from the master logger
    try:
        master_logger = logging.getLogger(TA_logger.master_logger_name)
        if master_logger.handlers:
            # Copy handlers from master logger
            for handler in master_logger.handlers:
                if handler not in logger.handlers:
                    logger.addHandler(handler)
    except:
        pass
    logger.propagate = True
    logger.setLevel(logging.INFO)
    return logger

# Create logger instance
logger = get_logger()

try:
    from cellpose import models, utils, io
    import cellpose
    CELLPOSE_AVAILABLE = True
    # Check Cellpose version for API compatibility
    try:
        CELLPOSE_VERSION = tuple(map(int, cellpose.__version__.split('.')))
    except:
        CELLPOSE_VERSION = (0, 0, 0)
except ImportError:
    CELLPOSE_AVAILABLE = False
    CELLPOSE_VERSION = (0, 0, 0)
    print("Warning: Cellpose not installed. Please install with: pip install cellpose")


def get_output_path(image_path):
    """
    Get the output path for outlines.tif based on input image path.
    
    Args:
        image_path: Path to input image
        
    Returns:
        str: Path to output outlines.tif file (cellpose-generated outlines)
    """
    filename_without_ext = os.path.splitext(image_path)[0]
    output_dir = filename_without_ext
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, 'outlines.tif')


def get_seg_npy_path(image_path):
    """
    Get the path to the _seg.npy file for a given image path.
    
    Args:
        image_path: Path to input image or handCorrection.tif
        
    Returns:
        str: Path to _seg.npy file, or None if not found
    """
    # If it's a handCorrection.tif or outlines.tif path, get the directory
    if 'handCorrection.tif' in image_path or 'outlines.tif' in image_path:
        output_dir = os.path.dirname(image_path)
        # Try to find _seg.npy in this directory
        seg_files = [f for f in os.listdir(output_dir) if f.endswith('_seg.npy')]
        if seg_files:
            return os.path.join(output_dir, seg_files[0])
        return None
    
    # Otherwise, construct path from image path
    filename_without_ext = os.path.splitext(image_path)[0]
    output_dir = filename_without_ext
    image_name = os.path.basename(filename_without_ext)
    seg_path = os.path.join(output_dir, f'{image_name}_seg.npy')
    
    if os.path.exists(seg_path):
        return seg_path
    return None


def load_seg_npy(seg_path):
    """
    Load and return the contents of a _seg.npy file.
    
    Args:
        seg_path: Path to _seg.npy file
        
    Returns:
        dict: Dictionary containing masks, flows, outlines, etc., or None if error
    """
    try:
        dat = np.load(seg_path, allow_pickle=True).item()
        return dat
    except Exception as e:
        logger.error(f'Failed to load _seg.npy file {seg_path}: {e}')
        return None


def inspect_seg_npy(seg_path):
    """
    Inspect and print information about a _seg.npy file.
    
    Args:
        seg_path: Path to _seg.npy file
        
    Returns:
        dict: Dictionary with inspection results
    """
    dat = load_seg_npy(seg_path)
    if dat is None:
        return None
    
    result = {
        'filename': dat.get('filename', 'Unknown'),
        'keys': list(dat.keys()),
        'masks_shape': None,
        'masks_unique_labels': None,
        'num_cells': None,
        'background_area': None,
        'small_background_regions': None
    }
    
    if 'masks' in dat:
        masks = dat['masks']
        result['masks_shape'] = masks.shape
        unique_labels = np.unique(masks)
        result['masks_unique_labels'] = len(unique_labels) - 1  # Exclude 0 (background)
        result['num_cells'] = len(unique_labels) - 1
        
        # Calculate background area
        background = (masks == 0)
        result['background_area'] = np.count_nonzero(background)
        
        # Find small background regions
        from skimage.measure import label, regionprops
        bg_labels = label(background, connectivity=2)
        bg_regions = regionprops(bg_labels)
        
        # Count small background regions (potential gaps)
        small_regions = [r for r in bg_regions if r.area <= 100]  # <= 100 pixels
        result['small_background_regions'] = {
            'count': len(small_regions),
            'total_area': sum(r.area for r in small_regions),
            'sizes': [r.area for r in small_regions[:10]]  # First 10 sizes
        }
    
    # Print inspection results
    print(f"\n=== Inspection of {seg_path} ===")
    print(f"Filename: {result['filename']}")
    print(f"Keys in file: {result['keys']}")
    if result['masks_shape']:
        print(f"Masks shape: {result['masks_shape']}")
        print(f"Number of cells: {result['num_cells']}")
        print(f"Background area (pixels): {result['background_area']}")
        if result['small_background_regions']:
            print(f"Small background regions (<=100 pixels): {result['small_background_regions']['count']}")
            print(f"Total area of small regions: {result['small_background_regions']['total_area']} pixels")
            if result['small_background_regions']['sizes']:
                print(f"Sample sizes: {result['small_background_regions']['sizes']}")
    print("=" * 50 + "\n")
    
    return result


def _call_progress_callback(progress_callback, value):
    """
    Helper function to call progress callback, handling both Qt Signals and regular callables.
    
    Args:
        progress_callback: Either a Qt Signal (with emit method) or a regular callable
        value: Progress value to pass to the callback
    """
    if progress_callback is None:
        return
    # Check if it's a Qt Signal (has emit method)
    if hasattr(progress_callback, 'emit'):
        progress_callback.emit(value)
    else:
        # Regular callable
        progress_callback(value)


def convert_to_handcorrection_format(masks):
    """
    Convert Cellpose label masks to outlines.tif format.
    
    The outlines format uses:
    - Outlines (boundaries) = 255 (white)
    - Background/cells = 0 (black)
    
    Args:
        masks: Label mask from Cellpose (0=background, 1-N=cell IDs)
        
    Returns:
        numpy.ndarray: Outline mask in outlines.tif format
    """
    # Extract outlines instead of binary mask
    return extract_outlines(masks, mode='boundaries')


def extract_outlines(masks, mode='boundaries'):
    """
    Extract cell outlines from labeled masks.
    
    Args:
        masks: Label mask from Cellpose (0=background, 1-N=cell IDs)
        mode: 'boundaries' (uses find_boundaries) or 'outlines' (Cellpose-style)
        
    Returns:
        numpy.ndarray: Outline mask (255=outlines, 0=background/cells) skeletonized to 1px width
    """
    if mode == 'boundaries':
        # Use skimage to find boundaries between regions
        boundaries = find_boundaries(masks, mode='inner', connectivity=1)
        # Convert to handCorrection-like format: boundaries=255, rest=0
        outline_mask = np.zeros_like(masks, dtype=np.uint8)
        outline_mask[boundaries] = 255
    else:
        # Alternative: use Cellpose's outline method if available
        try:
            from cellpose import utils
            outlines = utils.outlines_list(masks)
            outline_mask = np.zeros_like(masks, dtype=np.uint8)
            for outline in outlines:
                if len(outline) > 0:
                    outline_mask[outline[:, 0], outline[:, 1]] = 255
        except:
            # Fallback to boundaries
            return extract_outlines(masks, mode='boundaries')
    
    # Skeletonize the outline mask to 1px width
    # Convert to boolean for skeletonize (255 -> True, 0 -> False)
    outline_binary = (outline_mask == 255).astype(bool)
    # Apply skeletonization
    skeleton = skeletonize(outline_binary)
    # Convert back to uint8 format (True -> 255, False -> 0)
    outline_mask = np.zeros_like(masks, dtype=np.uint8)
    outline_mask[skeleton] = 255
    
    return outline_mask


def segment_image(image_path, model_type='cyto', diameter=None, 
                 channels=[0,0], flow_threshold=0.4, cellprob_threshold=0.0,
                 use_gpu=True, norm_percentiles=(1.0, 99.0), niter_dynamics=200,
                 progress_callback=None):
    """
    Segment image using Cellpose.
    
    Args:
        image_path: Path to input image
        model_type: 'cyto', 'cyto2', 'nuclei', or path to custom model
        diameter: Cell diameter in pixels (None for auto)
        channels: [channel1, channel2] for cyto models, [0,0] for grayscale
        flow_threshold: Flow error threshold (default 0.4)
        cellprob_threshold: Cell probability threshold (default 0.0)
        use_gpu: Whether to use GPU (default True)
        norm_percentiles: Tuple of (lower, upper) percentiles for normalization (default (1.0, 99.0))
        niter_dynamics: Number of iterations for dynamics (default 200)
        progress_callback: Optional callback for progress updates
        
    Returns:
        numpy.ndarray: Outline mask in outlines.tif format
    """
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose is not installed. Please install with: pip install cellpose")
    
    # Load image using existing Img class
    img = Img(image_path)
    
    # Handle multi-channel images
    if len(img.shape) == 3 and img.shape[2] > 1:
        # Multi-channel image - use specified channels
        if channels[0] == channels[1]:
            # Single channel mode
            img_data = img[..., channels[0]]
        else:
            # Two-channel mode
            img_data = np.stack([img[..., channels[0]], img[..., channels[1]]], axis=-1)
    elif len(img.shape) == 2:
        # Grayscale image
        img_data = img
    else:
        # Take first channel if 3D
        img_data = img[..., 0] if len(img.shape) == 3 else img
    
    # Ensure 2D or 2-channel for Cellpose
    if len(img_data.shape) == 3 and img_data.shape[2] == 1:
        img_data = img_data[..., 0]
    
    # Initialize Cellpose model
    _call_progress_callback(progress_callback, 10)
    
    # Set GPU preference
    gpu = use_gpu
    
    # Cellpose v4.0.1+ uses pretrained_model instead of model_type
    if CELLPOSE_VERSION >= (4, 0, 1):
        # In v4.0.1+, use pretrained_model parameter to avoid deprecation warnings
        try:
            # Try CellposeModel with pretrained_model first (most common API)
            model = models.CellposeModel(pretrained_model=model_type, gpu=gpu)
        except (TypeError, AttributeError):
            # Fallback: try Cellpose class if it exists
            try:
                import inspect
                if hasattr(models, 'Cellpose'):
                    sig = inspect.signature(models.Cellpose.__init__)
                    if 'pretrained_model' in sig.parameters:
                        model = models.Cellpose(pretrained_model=model_type, gpu=gpu)
                    else:
                        model = models.Cellpose(gpu=gpu)
                else:
                    model = models.CellposeModel(gpu=gpu)
            except:
                model = models.CellposeModel(gpu=gpu)
    else:
        # Older versions use model_type
        model = models.CellposeModel(model_type=model_type, gpu=gpu)
    
    _call_progress_callback(progress_callback, 30)
    
    # Run segmentation
    # Prepare eval arguments
    eval_kwargs = {
        'diameter': diameter,
        'flow_threshold': flow_threshold,
        'cellprob_threshold': cellprob_threshold,
        'normalize': True
    }
    
    # Add normalization percentiles if supported
    # In Cellpose v4.0.1+, normalize can be a bool or a dict with 'lower' and 'upper' keys
    if CELLPOSE_VERSION >= (4, 0, 1):
        # Convert tuple to dict format for newer versions
        try:
            eval_kwargs['normalize'] = {
                'lower': norm_percentiles[0],
                'upper': norm_percentiles[1]
            }
        except:
            eval_kwargs['normalize'] = True
    else:
        # For older versions, use boolean
        eval_kwargs['normalize'] = True
    
    # Add niter_dynamics if supported
    try:
        import inspect
        sig = inspect.signature(model.eval)
        if 'niter' in sig.parameters or 'niter_dynamics' in sig.parameters:
            param_name = 'niter_dynamics' if 'niter_dynamics' in sig.parameters else 'niter'
            eval_kwargs[param_name] = niter_dynamics
    except:
        pass
    
    # channels parameter is deprecated in v4.0.1+, only include for older versions
    if CELLPOSE_VERSION < (4, 0, 1):
        eval_kwargs['channels'] = channels if len(img_data.shape) == 3 else [0, 0]
    
    result = model.eval(img_data, **eval_kwargs)
    
    # Handle different return values: v4.0.1+ returns 3 values, older versions return 4
    if len(result) == 3:
        masks, flows, styles = result
        diams = None  # Not returned in newer versions
    else:
        masks, flows, styles, diams = result
    
    _call_progress_callback(progress_callback, 80)
    
    # Save using Cellpose's io.save_masks
    # io.save_masks saves in the same directory as the images
    output_dir = os.path.dirname(get_output_path(image_path))
    image_name = os.path.basename(os.path.splitext(image_path)[0])
    
    # Change to output directory temporarily to save masks there
    original_cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        # Removed io.save_masks call to prevent cp_masks.tif from being created during fresh runs
        # io.save_masks(
        #     images=[img_data],
        #     masks=[masks],
        #     flows=[flows],
        #     file_names=[image_name],  # Note: file_names is plural, not image_names
        #     png=False,  # Save as .tif
        #     tif=True
        # )
        
        # Save _seg.npy file for later use in gap filling
        # File will be saved as {image_name}_seg.npy in output_dir
        try:
            img_channels = channels if len(img_data.shape) == 3 else [0, 0]
            io.masks_flows_to_seg(
                images=[img_data],
                masks=[masks],
                flows=[flows],
                file_names=[image_name],  # Note: file_names is plural and expects a list
                channels=img_channels
            )
            seg_file_path = os.path.join(output_dir, f'{image_name}_seg.npy')
            if os.path.exists(seg_file_path):
                print(f'[CELLPOSE] Saved _seg.npy file: {seg_file_path}')
                logger.info(f'Saved _seg.npy file: {seg_file_path}')
            else:
                print(f'[CELLPOSE WARNING] _seg.npy file not found at: {seg_file_path}')
                logger.warning(f'_seg.npy file not found at expected path: {seg_file_path}')
        except Exception as e:
            print(f'[CELLPOSE ERROR] Failed to save _seg.npy file: {e}')
            logger.error(f'Failed to save _seg.npy file: {e}')
            import traceback
            traceback.print_exc()
    finally:
        os.chdir(original_cwd)
    
    # Save outlines as outlines.tif format (from cellpose)
    outlines = extract_outlines(masks, mode='boundaries')
    output_path = get_output_path(image_path)
    
    # Fill holes automatically at the end of segmentation
    _call_progress_callback(progress_callback, 90)
    logger.info('Filling holes in segmentation result...')
    
    # Try to use _seg.npy file if available (more accurate)
    filled_outlines = None
    seg_file_path = os.path.join(output_dir, f'{image_name}_seg.npy')
    if os.path.exists(seg_file_path):
        logger.info(f'Using _seg.npy file for hole filling: {seg_file_path}')
        filled_outlines = fill_holes_using_seg_npy(seg_file_path, max_region_size=500)
        if filled_outlines is not None:
            logger.info('Successfully filled holes using _seg.npy data')
    
    # Fall back to outline-based method if _seg.npy not available or failed
    if filled_outlines is None:
        logger.info('Using outline-based hole filling method')
        filled_outlines = fill_holes_in_outline_mask(
            outlines, 
            max_hole_size=None,  # Fill all holes
            extend_membranes=True, 
            extension_radius=1
        )
    
    # Save the filled outlines
    outline_img = Img(filled_outlines)
    outline_img.save(output_path)
    logger.info(f'Saved filled outlines to: {output_path}')
    
    _call_progress_callback(progress_callback, 100)
    
    return filled_outlines


def segment_batch(image_list, model_type='cyto', diameter=None, 
                 channels=[0,0], flow_threshold=0.4, cellprob_threshold=0.0,
                 use_gpu=True, norm_percentiles=(1.0, 99.0), niter_dynamics=200,
                 progress_callback=None, batch_size=4):
    """
    Process multiple images with Cellpose using native batch processing.
    
    Args:
        image_list: List of image paths to segment
        model_type: 'cyto', 'cyto2', 'nuclei', or path to custom model
        diameter: Cell diameter in pixels (None for auto)
        channels: [channel1, channel2] for cyto models
        flow_threshold: Flow error threshold
        cellprob_threshold: Cell probability threshold
        use_gpu: Whether to use GPU (default True)
        norm_percentiles: Tuple of (lower, upper) percentiles for normalization (default (1.0, 99.0))
        niter_dynamics: Number of iterations for dynamics (default 200)
        progress_callback: Optional callback for progress updates
        batch_size: Number of images to process simultaneously (default 4)
        
    Returns:
        list: List of output mask paths
    """
    # Ensure logger has GUI handler (refresh it)
    logger = get_logger()
    
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose is not installed. Please install with: pip install cellpose")
    
    # Print to stdout - should appear in GUI log box
    print(f'[CELLPOSE] Starting batch segmentation: {len(image_list)} images with batch_size={batch_size}')
    logger.info(f'Starting Cellpose batch segmentation: {len(image_list)} images with batch_size={batch_size}')
    
    output_paths = []
    total = len(image_list)
    
    # Initialize model once for batch processing
    # Set GPU preference
    gpu = use_gpu
    
    # Cellpose v4.0.1+ uses pretrained_model instead of model_type
    if CELLPOSE_VERSION >= (4, 0, 1):
        # In v4.0.1+, use pretrained_model parameter to avoid deprecation warnings
        try:
            # Try CellposeModel with pretrained_model first (most common API)
            model = models.CellposeModel(pretrained_model=model_type, gpu=gpu)
        except (TypeError, AttributeError):
            # Fallback: try Cellpose class if it exists
            try:
                import inspect
                if hasattr(models, 'Cellpose'):
                    sig = inspect.signature(models.Cellpose.__init__)
                    if 'pretrained_model' in sig.parameters:
                        model = models.Cellpose(pretrained_model=model_type, gpu=gpu)
                    else:
                        model = models.Cellpose(gpu=gpu)
                else:
                    model = models.CellposeModel(gpu=gpu)
            except:
                model = models.CellposeModel(gpu=gpu)
    else:
        # Older versions use model_type
        model = models.CellposeModel(model_type=model_type, gpu=gpu)
    
    # Load all images first
    _call_progress_callback(progress_callback, 10)
    logger.info('Loading images for batch processing...')
    
    all_images_data = []
    all_image_names = []
    all_img_paths = []
    
    for i, img_path in enumerate(image_list):
        # Calculate progress: 10-40% for loading
        progress = 10 + int((i / total) * 30)
        _call_progress_callback(progress_callback, progress)
        
        try:
            # Load image
            img = Img(img_path)
            
            # Handle multi-channel images
            if len(img.shape) == 3 and img.shape[2] > 1:
                if channels[0] == channels[1]:
                    img_data = img[..., channels[0]]
                else:
                    img_data = np.stack([img[..., channels[0]], img[..., channels[1]]], axis=-1)
            elif len(img.shape) == 2:
                img_data = img
            else:
                img_data = img[..., 0] if len(img.shape) == 3 else img
            
            if len(img_data.shape) == 3 and img_data.shape[2] == 1:
                img_data = img_data[..., 0]
            
            all_images_data.append(img_data)
            image_name = os.path.basename(os.path.splitext(img_path)[0])
            all_image_names.append(image_name)
            all_img_paths.append(img_path)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_images_data:
        logger.error('No images loaded successfully')
        return []
    
    # Prepare eval arguments
    eval_kwargs = {
        'diameter': diameter,
        'flow_threshold': flow_threshold,
        'cellprob_threshold': cellprob_threshold,
        'normalize': True,
        'batch_size': batch_size
    }
    
    # Add normalization percentiles if supported
    # In Cellpose v4.0.1+, normalize can be a bool or a dict with 'lower' and 'upper' keys
    if CELLPOSE_VERSION >= (4, 0, 1):
        # Convert tuple to dict format for newer versions
        try:
            eval_kwargs['normalize'] = {
                'lower': norm_percentiles[0],
                'upper': norm_percentiles[1]
            }
        except:
            eval_kwargs['normalize'] = True
    else:
        # For older versions, use boolean
        eval_kwargs['normalize'] = True
    
    # Add niter_dynamics if supported
    try:
        import inspect
        sig = inspect.signature(model.eval)
        if 'niter' in sig.parameters or 'niter_dynamics' in sig.parameters:
            param_name = 'niter_dynamics' if 'niter_dynamics' in sig.parameters else 'niter'
            eval_kwargs[param_name] = niter_dynamics
    except:
        pass
    
    # channels parameter is deprecated in v4.0.1+, only include for older versions
    if CELLPOSE_VERSION < (4, 0, 1):
        # For batch processing, we need to determine channels from the first image
        first_img_shape = all_images_data[0].shape
        eval_kwargs['channels'] = channels if len(first_img_shape) == 3 else [0, 0]
    
    # Run batch segmentation - process all images at once
    _call_progress_callback(progress_callback, 40)
    logger.info(f'Processing {len(all_images_data)} images in batch mode...')
    
    try:
        result = model.eval(all_images_data, **eval_kwargs)
    except Exception as e:
        logger.error(f'Batch segmentation failed: {e}')
        import traceback
        traceback.print_exc()
        # Fall back to sequential processing if batch fails
        logger.info('Falling back to sequential processing...')
        return segment_batch(image_list, model_type=model_type, diameter=diameter,
                           channels=channels, flow_threshold=flow_threshold,
                           cellprob_threshold=cellprob_threshold, use_gpu=use_gpu,
                           norm_percentiles=norm_percentiles, niter_dynamics=niter_dynamics,
                           progress_callback=progress_callback, batch_size=1)
    
    # Handle different return values: v4.0.1+ returns 3 values, older versions return 4
    if len(result) == 3:
        all_masks, all_flows, all_styles = result
        all_diams = None  # Not returned in newer versions
    else:
        all_masks, all_flows, all_styles, all_diams = result
    
    # Ensure results are lists (batch processing should return lists)
    if not isinstance(all_masks, list):
        all_masks = [all_masks]
    if not isinstance(all_flows, list):
        all_flows = [all_flows]
    
    # Verify we have the right number of results
    if len(all_masks) != len(all_images_data):
        logger.warning(f'Mismatch: {len(all_masks)} masks for {len(all_images_data)} images')
        # Truncate or pad as needed
        min_len = min(len(all_masks), len(all_images_data))
        all_masks = all_masks[:min_len]
        all_flows = all_flows[:min_len]
        all_images_data = all_images_data[:min_len]
        all_image_names = all_image_names[:min_len]
        all_img_paths = all_img_paths[:min_len]
    
    _call_progress_callback(progress_callback, 70)
    
    # Save all masks using Cellpose's io.save_masks
    # Note: io.save_masks saves in the directory of each image
    if all_masks:
        _call_progress_callback(progress_callback, 80)
        try:
            # Save masks for each image in its respective output directory
            for img_path, img_data, masks, flows, image_name in zip(
                all_img_paths, all_images_data, all_masks, all_flows, all_image_names
            ):
                output_dir = os.path.dirname(get_output_path(img_path))
                original_cwd = os.getcwd()
                try:
                    os.chdir(output_dir)
                    # Removed io.save_masks call to prevent cp_masks.tif from being created during fresh runs
                    # io.save_masks(
                    #     images=[img_data],
                    #     masks=[masks],
                    #     flows=[flows],
                    #     file_names=[image_name],  # Note: file_names is plural, not image_names
                    #     png=False,  # Save as .tif
                    #     tif=True
                    # )
                    
                    # Save _seg.npy file for later use in gap filling
                    # File will be saved as {image_name}_seg.npy in output_dir
                    try:
                        img_channels = channels if len(img_data.shape) == 3 else [0, 0]
                        io.masks_flows_to_seg(
                            images=[img_data],
                            masks=[masks],
                            flows=[flows],
                            file_names=[image_name],  # Note: file_names is plural and expects a list
                            channels=img_channels
                        )
                        seg_file_path = os.path.join(output_dir, f'{image_name}_seg.npy')
                        if os.path.exists(seg_file_path):
                            print(f'[CELLPOSE] Saved _seg.npy file: {seg_file_path}')
                            logger.info(f'Saved _seg.npy file: {seg_file_path}')
                        else:
                            print(f'[CELLPOSE WARNING] _seg.npy file not found at: {seg_file_path}')
                            logger.warning(f'_seg.npy file not found at expected path: {seg_file_path}')
                    except Exception as e:
                        print(f'[CELLPOSE WARNING] Failed to save _seg.npy file for {image_name}: {e}')
                        logger.error(f'Failed to save _seg.npy file for {image_name}: {e}')
                        import traceback
                        traceback.print_exc()
                finally:
                    os.chdir(original_cwd)
        except Exception as e:
            print(f"Error saving masks with io.save_masks: {e}")
            import traceback
            traceback.print_exc()
        
        # Save outlines as outlines.tif format (from cellpose)
        _call_progress_callback(progress_callback, 90)
        for img_path, masks, image_name in zip(all_img_paths, all_masks, all_image_names):
            try:
                outlines = extract_outlines(masks, mode='boundaries')
                output_path = get_output_path(img_path)
                output_dir = os.path.dirname(output_path)
                
                # Fill holes automatically at the end of segmentation
                logger.info(f'Filling holes in segmentation result for {image_name}...')
                
                # Try to use _seg.npy file if available (more accurate)
                filled_outlines = None
                seg_file_path = os.path.join(output_dir, f'{image_name}_seg.npy')
                if os.path.exists(seg_file_path):
                    logger.info(f'Using _seg.npy file for hole filling: {seg_file_path}')
                    filled_outlines = fill_holes_using_seg_npy(seg_file_path, max_region_size=500)
                    if filled_outlines is not None:
                        logger.info('Successfully filled holes using _seg.npy data')
                
                # Fall back to outline-based method if _seg.npy not available or failed
                if filled_outlines is None:
                    logger.info('Using outline-based hole filling method')
                    filled_outlines = fill_holes_in_outline_mask(
                        outlines, 
                        max_hole_size=None,  # Fill all holes
                        extend_membranes=True, 
                        extension_radius=1
                    )
                
                # Save the filled outlines
                outline_img = Img(filled_outlines)
                outline_img.save(output_path)
                logger.info(f'Saved filled outlines to: {output_path}')
                output_paths.append(output_path)
            except Exception as e:
                print(f"Error saving handCorrection format for {img_path}: {e}")
                import traceback
                traceback.print_exc()
    
    _call_progress_callback(progress_callback, 100)
    
    return output_paths


def fill_holes_in_outline_mask(outline_mask, max_hole_size=None, extend_membranes=True, extension_radius=1):
    """
    Fill holes in outline mask for epithelial tissue.
    
    In epithelia, there should be no gaps between cells. This function:
    1. Converts outline mask to labeled regions
    2. Fills holes within each cell region
    3. Extends membranes to fill small gaps between cells
    4. Converts back to outline format
    
    Args:
        outline_mask: Outline mask (255=boundaries, 0=background/cells)
        max_hole_size: Maximum hole size to fill in pixels (None = fill all holes)
        extend_membranes: Whether to extend membranes slightly to fill small gaps
        extension_radius: Radius for membrane extension (default 1 pixel)
        
    Returns:
        numpy.ndarray: Outline mask with holes filled (255=boundaries, 0=background/cells)
    """
    from scipy.ndimage import binary_fill_holes, binary_closing, distance_transform_edt
    
    if outline_mask is None or outline_mask.size == 0:
        return outline_mask
    
    # Convert outline mask to labeled regions
    # In outline format: 255 = boundaries, 0 = background/cells
    # We need to invert this: boundaries become 0, cells become non-zero
    # Use label with background=255 to treat boundaries as background
    lab_mask = label(outline_mask, connectivity=1, background=255)
    
    # Get unique cell labels (excluding background 0)
    unique_labels = np.unique(lab_mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) == 0:
        logger.warning('fill_holes_in_outline_mask: No cell regions found in mask')
        return outline_mask
    
    logger.info(f'fill_holes_in_outline_mask: Processing {len(unique_labels)} cell regions')
    
    filled_mask = lab_mask.copy()
    
    # Fill holes within each cell region
    for cell_id in unique_labels:
        # Create binary mask for this cell
        cell_mask = (filled_mask == cell_id).astype(bool)
        
        # Fill holes in this cell
        if max_hole_size is not None:
            # Use skimage's remove_small_holes for size-limited filling
            cell_mask = remove_small_holes(cell_mask, area_threshold=max_hole_size)
        else:
            # Fill all holes using scipy
            cell_mask = binary_fill_holes(cell_mask)
        
        # Update the mask
        filled_mask[cell_mask] = cell_id
    
    # Extend membranes to fill small gaps between cells using morphological closing
    if extend_membranes and extension_radius > 0:
        # Create binary mask of all cells
        all_cells = (filled_mask > 0).astype(bool)
        
        # Use morphological closing to fill small gaps
        # Closing = dilation followed by erosion
        structure_size = 2 * extension_radius + 1
        structure = np.ones((structure_size, structure_size), dtype=bool)
        
        # Apply closing to fill gaps
        closed_cells = binary_closing(all_cells, structure=structure)
        
        # Find gap regions that were filled
        gaps_filled = closed_cells & ~all_cells
        
        if np.any(gaps_filled):
            # Identify border cells (cells touching image edges) - these should not be modified
            border_cells = set()
            height, width = filled_mask.shape
            
            # Check top and bottom edges
            for x in range(width):
                top_label = filled_mask[0, x]
                bottom_label = filled_mask[height - 1, x]
                if top_label > 0:
                    border_cells.add(top_label)
                if bottom_label > 0:
                    border_cells.add(bottom_label)
            
            # Check left and right edges
            for y in range(height):
                left_label = filled_mask[y, 0]
                right_label = filled_mask[y, width - 1]
                if left_label > 0:
                    border_cells.add(left_label)
                if right_label > 0:
                    border_cells.add(right_label)
            
            logger.info(f'fill_holes_in_outline_mask: Found {len(border_cells)} border cells that will be preserved')
            
            # Use watershed on distance transform to assign gaps to nearest cells
            # This is more efficient and accurate than pixel-by-pixel search
            
            # Create distance transform from cell boundaries
            # Distance is 0 at cell boundaries, increases into gaps
            distance_from_cells = distance_transform_edt(~all_cells)
            
            # Create mask for watershed: include both original cells and gaps to fill
            watershed_mask = closed_cells
            
            # Use filled_mask as markers (each cell is a marker with its label)
            # Watershed will assign gap pixels to nearest cell based on distance
            # Invert distance so watershed finds minima (closest to cells)
            filled_mask_result = watershed(-distance_from_cells, filled_mask, mask=watershed_mask)
            
            # Preserve border cells: restore original values for border cell pixels
            # This ensures border cells are not modified by gap filling
            if border_cells:
                border_mask = np.zeros_like(filled_mask, dtype=bool)
                for border_label in border_cells:
                    border_mask[filled_mask == border_label] = True
                
                # Restore original border cell pixels
                filled_mask_result[border_mask] = filled_mask[border_mask]
            
            # Only update pixels that were actually gaps (not original cell pixels)
            # This prevents modifying existing cells
            gap_mask = gaps_filled
            filled_mask[gap_mask] = filled_mask_result[gap_mask]
            
            logger.info(f'fill_holes_in_outline_mask: Extended membranes to fill {np.count_nonzero(gaps_filled)} gap pixels using watershed')
    
    # Convert back to outline format
    boundaries = find_boundaries(filled_mask, mode='inner', connectivity=1)
    result_outline = np.zeros_like(outline_mask, dtype=np.uint8)
    result_outline[boundaries] = 255
    
    # Skeletonize the outline mask to 1px width
    # Convert to boolean for skeletonize (255 -> True, 0 -> False)
    outline_binary = (result_outline == 255).astype(bool)
    # Apply skeletonization
    skeleton = skeletonize(outline_binary)
    # Convert back to uint8 format (True -> 255, False -> 0)
    result_outline = np.zeros_like(outline_mask, dtype=np.uint8)
    result_outline[skeleton] = 255
    
    return result_outline


def fill_holes_using_seg_npy(seg_npy_path, max_region_size=500):
    """
    Fill holes by merging small background regions from _seg.npy file into nearest cells.
    This uses the original Cellpose labeled masks for more accurate gap detection.
    
    Args:
        seg_npy_path: Path to _seg.npy file
        max_region_size: Maximum size (in pixels) of background regions to merge (default: 500)
        
    Returns:
        numpy.ndarray: Outline mask (outlines.tif format) with holes filled, or None if error
    """
    from skimage.measure import regionprops
    from scipy.ndimage import distance_transform_edt
    
    # Load the _seg.npy file
    dat = load_seg_npy(seg_npy_path)
    if dat is None:
        logger.warning(f'Could not load _seg.npy file: {seg_npy_path}')
        return None
    
    if 'masks' not in dat:
        logger.warning('No masks found in _seg.npy file')
        return None
    
    masks = dat['masks']
    logger.info(f'Filling holes using _seg.npy: {len(np.unique(masks)) - 1} cells detected')
    
    # Identify background (label 0)
    background = (masks == 0)
    background_area = np.count_nonzero(background)
    
    # Find all background regions
    bg_labels = label(background, connectivity=2)
    bg_regions = regionprops(bg_labels)
    
    # Filter small regions to merge
    small_regions = [r for r in bg_regions if r.area <= max_region_size]
    logger.info(f'Found {len(small_regions)} small background regions (<= {max_region_size} pixels) to merge')
    
    if len(small_regions) == 0:
        logger.info('No small regions to merge')
        # Still convert to outline format
        boundaries = find_boundaries(masks, mode='inner', connectivity=1)
        outline_mask = np.zeros_like(masks, dtype=np.uint8)
        outline_mask[boundaries] = 255
        # Skeletonize the outline mask to 1px width
        outline_binary = (outline_mask == 255).astype(bool)
        skeleton = skeletonize(outline_binary)
        outline_mask = np.zeros_like(masks, dtype=np.uint8)
        outline_mask[skeleton] = 255
        return outline_mask
    
    # Create mask of small regions
    small_regions_mask = np.zeros_like(masks, dtype=bool)
    for region in small_regions:
        coords = region.coords
        for coord in coords:
            small_regions_mask[coord[0], coord[1]] = True
    
    # Create binary mask of all cells
    all_cells = (masks > 0).astype(bool)
    
    # Use distance transform to find nearest cell for each small region pixel
    distance_from_cells = distance_transform_edt(~all_cells)
    
    # Create mask for watershed: include both cells and small regions to merge
    watershed_mask = all_cells | small_regions_mask
    
    # Use watershed to assign small region pixels to nearest cells
    merged_masks = watershed(-distance_from_cells, masks, mask=watershed_mask)
    
    # Count how many pixels were merged
    merged_pixels = np.count_nonzero((merged_masks > 0) & (masks == 0))
    logger.info(f'Merged {merged_pixels} background pixels into cells')
    
    # Convert merged labeled mask to outline format
    boundaries = find_boundaries(merged_masks, mode='inner', connectivity=1)
    outline_mask = np.zeros_like(merged_masks, dtype=np.uint8)
    outline_mask[boundaries] = 255
    
    # Skeletonize the outline mask to 1px width
    # Convert to boolean for skeletonize (255 -> True, 0 -> False)
    outline_binary = (outline_mask == 255).astype(bool)
    # Apply skeletonization
    skeleton = skeletonize(outline_binary)
    # Convert back to uint8 format (True -> 255, False -> 0)
    outline_mask = np.zeros_like(merged_masks, dtype=np.uint8)
    outline_mask[skeleton] = 255
    
    return outline_mask

