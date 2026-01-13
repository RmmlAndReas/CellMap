"""Image registration for tracking.

This module provides image registration functions for aligning images between
time frames to improve tracking accuracy. Uses phase cross-correlation for
translation estimation.
"""

from skimage.registration import phase_cross_correlation
from skimage import transform
import numpy as np
import json
import os
from pathlib import Path

__DEBUG__ = False

# #region agent log
def _debug_log(location, message, data, hypothesis_id=None):
    log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
    try:
        with open(log_path, 'a') as f:
            import time
            log_entry = {
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000)
            }
            f.write(json.dumps(log_entry) + "\n")
    except:
        pass
# #endregion

def apply_translation(img, y, x):
    """
    Applies translation to an image.

    Args:
        img (numpy.ndarray): The input image.
        y (float): Translation along the y-axis.
        x (float): Translation along the x-axis.

    Returns:
        numpy.ndarray: The translated image.
    """
    afine_tf = transform.AffineTransform(translation=(x, y))
    translated = transform.warp(img, inverse_map=afine_tf, order=0, preserve_range=True)
    return translated

def pre_register_images(orig_t0, orig_t1, apply_to_orig_t0=False):
    """
    Performs pre-registration between two images.

    Args:
        orig_t0 (numpy.ndarray): The first image.
        orig_t1 (numpy.ndarray): The second image.
        apply_to_orig_t0 (bool, optional): Whether to apply the translation to orig_t0. Defaults to False.

    Returns:
        tuple or numpy.ndarray: If apply_to_orig_t0 is False, returns the global translations (gloabl_t0, gloabl_t1).
        If apply_to_orig_t0 is True, returns the translated orig_t0 image.

    Notes:
        - The function uses phase cross-correlation to estimate the global translation between the two images.
        - The global translation is then applied to either orig_t0 or both orig_t0 and orig_t1, depending on the
          value of apply_to_orig_t0.
    """
    global_translation, error, diffphase = phase_cross_correlation(orig_t0, orig_t1)
    gloabl_t0 = -global_translation[0]
    gloabl_t1 = -global_translation[1]
    if not apply_to_orig_t0:
        return gloabl_t0, gloabl_t1
    else:
        return apply_translation(orig_t0, -gloabl_t0, -gloabl_t1)

def get_pyramidal_registration(orig_t0, orig_t1, depth=2, threshold_translation=20,
                                landmarks_t0=None, landmarks_t1=None):
    """
    Performs pyramidal registration between two images.

    Args:
        orig_t0 (numpy.ndarray): The first image.
        orig_t1 (numpy.ndarray): The second image.
        depth (int, optional): The depth of the pyramid. Defaults to 2.
        threshold_translation (float, optional): The threshold for ignoring large translations. Defaults to 20.
        landmarks_t0 (list, optional): List of (y, x) tuples for landmarks in frame t0. Defaults to None.
        landmarks_t1 (list, optional): List of (y, x) tuples for landmarks in frame t1. Defaults to None.

    Returns:
        numpy.ndarray: The translation matrix representing the translations for each pixel.

    Notes:
        - If landmarks are provided (>= 3 points), uses them to initialize the translation matrix.
        - Otherwise, applies pre-registration between orig_t0 and orig_t1.
        - It then iteratively performs registration on blocks of decreasing size.
        - Translations are accumulated in the translation matrix.
    """

    translation_matrix = np.zeros((*orig_t0.shape, 2))

    # Use landmarks if provided and sufficient
    if landmarks_t0 and landmarks_t1 and len(landmarks_t0) >= 3 and len(landmarks_t0) == len(landmarks_t1):
        # Landmark-based initialization
        src = np.array(landmarks_t0)
        dst = np.array(landmarks_t1)
        
        # #region agent log
        _debug_log("registration.py:82", "Landmarks check passed", {
            "num_landmarks": len(landmarks_t0),
            "src_shape": str(src.shape),
            "dst_shape": str(dst.shape)
        }, "A")
        # #endregion
        
        # #region agent log
        try:
            import skimage
            skimage_version = skimage.__version__
        except:
            skimage_version = "unknown"
        _debug_log("registration.py:87", "Checking scikit-image version and transform methods", {
            "skimage_version": skimage_version,
            "has_AffineTransform": hasattr(transform, "AffineTransform"),
            "has_PiecewiseAffineTransform": hasattr(transform, "PiecewiseAffineTransform"),
            "AffineTransform_methods": dir(transform.AffineTransform) if hasattr(transform, "AffineTransform") else [],
            "PiecewiseAffineTransform_methods": dir(transform.PiecewiseAffineTransform) if hasattr(transform, "PiecewiseAffineTransform") else []
        }, "A,B,C,E")
        # #endregion
        
        # Estimate transformation
        if len(landmarks_t0) == 3:
            # #region agent log
            _debug_log("registration.py:88", "Using AffineTransform path (3 points)", {
                "num_points": 3
            }, "B")
            # #endregion
            # Use affine transform for exactly 3 points
            # #region agent log
            _debug_log("registration.py:90", "Before AffineTransform.estimate call", {
                "src": src.tolist(),
                "dst": dst.tolist()
            }, "B")
            # #endregion
            try:
                tform = transform.AffineTransform()
                success = tform.estimate(src, dst)
                # #region agent log
                _debug_log("registration.py:90", "AffineTransform.estimate completed", {
                    "tform_type": str(type(tform)),
                    "estimate_success": success,
                    "tform_not_none": tform is not None
                }, "B")
                # #endregion
                if not success:
                    tform = None
            except Exception as e:
                # #region agent log
                _debug_log("registration.py:90", "AffineTransform.estimate failed", {
                    "error_type": str(type(e).__name__),
                    "error_message": str(e)
                }, "B")
                # #endregion
                raise
        else:
            # #region agent log
            _debug_log("registration.py:92", "Using PiecewiseAffineTransform path (4+ points)", {
                "num_points": len(landmarks_t0)
            }, "A,C,D")
            # #endregion
            # Use piecewise-affine for 4+ points
            # #region agent log
            _debug_log("registration.py:93", "Before PiecewiseAffineTransform.estimate call", {
                "src": src.tolist(),
                "dst": dst.tolist()
            }, "A,C,D")
            # #endregion
            try:
                tform = transform.PiecewiseAffineTransform()
                success = tform.estimate(src, dst)
                # #region agent log
                _debug_log("registration.py:93", "PiecewiseAffineTransform.estimate completed", {
                    "tform_type": str(type(tform)),
                    "estimate_success": success,
                    "tform_not_none": tform is not None
                }, "A")
                # #endregion
                if not success:
                    tform = None
            except Exception as e:
                # #region agent log
                _debug_log("registration.py:93", "PiecewiseAffineTransform.estimate failed", {
                    "error_type": str(type(e).__name__),
                    "error_message": str(e),
                    "available_methods": [m for m in dir(transform.PiecewiseAffineTransform) if not m.startswith("_")]
                }, "A,C,D")
                # #endregion
                raise
        
        # Check if estimation succeeded
        if not tform:
            # Fall back to standard pre-registration if transform estimation failed
            gloabl_t0, gloabl_t1 = pre_register_images(orig_t0, orig_t1)
            translation_matrix[..., 0] += gloabl_t0
            translation_matrix[..., 1] += gloabl_t1
        else:
            # Create coordinate grid
            h, w = orig_t0.shape[:2]
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            coords_t0 = np.dstack([y_coords, x_coords])
            
            # Transform coordinates
            coords_t1 = tform(coords_t0.reshape(-1, 2)).reshape(h, w, 2)
            
            # Compute translation vectors
            translation_matrix[..., 0] = coords_t1[..., 0] - coords_t0[..., 0]  # dy
            translation_matrix[..., 1] = coords_t1[..., 1] - coords_t0[..., 1]  # dx
    else:
        # Standard pre-registration (backward compatible)
        gloabl_t0, gloabl_t1 = pre_register_images(orig_t0, orig_t1)
        translation_matrix[..., 0] += gloabl_t0
        translation_matrix[..., 1] += gloabl_t1

    height = orig_t0.shape[0]
    width = orig_t0.shape[1]

    block_size = 256

    for i in range(depth):
        try:
            tile_height = block_size // (i * 2)
            tile_width = block_size // (i * 2)
        except:
            tile_height = block_size
            tile_width = block_size

        for y in range(0, height, tile_height):
            for x in range(0, width, tile_width):
                afine_tf = transform.AffineTransform(
                    translation=(-translation_matrix[y, x, 1], -translation_matrix[y, x, 0]))
                translated = transform.warp(orig_t0, inverse_map=afine_tf, order=0, preserve_range=True)

                tile_t0 = translated[y:y + tile_height, x:x + tile_width]
                tile_t1 = orig_t1[y:y + tile_height, x:x + tile_width]

                shift, error, diffphase = phase_cross_correlation(tile_t0, tile_t1)
                t0 = -shift[0]
                t1 = -shift[1]

                if abs(t0) > threshold_translation or abs(t1) > threshold_translation:
                    continue

                translation_matrix[y:y + tile_height, x:x + tile_width, 0] += t0
                translation_matrix[y:y + tile_height, x:x + tile_width, 1] += t1

    return translation_matrix

