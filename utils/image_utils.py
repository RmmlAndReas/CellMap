"""Image utility functions for CellMap.

This module provides utility functions for image processing and conversion.
"""
import os
import numpy as np
import warnings
from PIL import Image
import tifffile
from skimage.util import img_as_ubyte
from scipy import ndimage as ndi
from utils.logger import TA_logger
from utils.image_io import Img

logger = TA_logger()


def _create_dir(filename):
    """Create directory if it doesn't exist."""
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def RGB_to_int24(RGBimg):
    """Convert a 3-channel RGB image to a single-channel int24 image."""
    RGB24 = (RGBimg[..., 0].astype(np.uint32) << 16) | (RGBimg[..., 1].astype(np.uint32) << 8) | RGBimg[..., 2].astype(np.uint32)
    return RGB24


def int24_to_RGB(RGB24):
    """Convert a 24-bit integer image to an RGB image."""
    RGBimg = np.zeros(shape=(*RGB24.shape, 3), dtype=np.uint8)
    for c in range(RGBimg.shape[-1]):
        RGBimg[..., c] = (RGB24 >> ((RGBimg.shape[-1] - c - 1) * 8)) & 0xFF
    return RGBimg


def is_binary(image):
    """Check if an image is binary (contains only two unique pixel values)."""
    mx = image.max()
    mn = image.min()
    if mx == mn:
        return True
    binary_pixels = (image == mn) | (image == mx)
    num_binary_pixels = np.count_nonzero(binary_pixels)
    if num_binary_pixels == image.size:
        return True
    return False


def auto_scale(img, individual_channels=True, min_px_count_in_percent=0.005):
    """Auto-normalize the pixel intensities of an image based on their distribution."""
    if not isinstance(img, np.ndarray):
        return img
    if len(img.shape) > 2 and individual_channels:
        if img.dtype != float:
            img = img.astype(float)
        for ch in range(img.shape[-1]):
            img[..., ch] = auto_scale(img[..., ch])
    else:
        if img.dtype != float:
            img = img.astype(float)
        mn = img.min()
        mx = img.max()
        if mn == mx:
            if mx != 0:
                img = img / mx * 255
            return img.astype(np.uint8)
        img = (img - mn) / (mx - mn) * 255
    return img.astype(np.uint8)


def _get_white_bounds(img):
    """Get bounding box of white (non-zero) pixels in an image."""
    coords = np.where(img != 0)
    if coords[0].size == 0:
        return None
    return np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1])


def get_white_bounds(imgs):
    """Get bounding box of white pixels, handling single images or lists."""
    if isinstance(imgs, list):
        bounds = [10000000, 0, 10000000, 0]
        for img in imgs:
            curbounds = _get_white_bounds(img)
            if curbounds is None:
                continue
            bounds[0] = min(curbounds[0], bounds[0])
            bounds[1] = max(curbounds[1], bounds[1])
            bounds[2] = min(curbounds[2], bounds[2])
            bounds[3] = max(curbounds[3], bounds[3])
        if bounds[0] == 10000000:
            return None
    else:
        bounds = _get_white_bounds(imgs)
    return bounds


def save_as_tiff(img, output_name, print_file_name=False, ijmetadata='copy', mode='IJ'):
    """Save an image as a TIFF file."""
    if print_file_name:
        print('saving', output_name)
    if output_name is None:
        logger.error("No output name specified")
        return
    _create_dir(output_name)
    if mode != 'IJ':
        tifffile.imwrite(output_name, img)
    else:
        out = img.copy()
        if out.dtype == np.int32:
            out = out.astype(np.float32)
        if out.dtype == np.int64:
            out = out.astype(np.float64)
        if out.dtype == bool:
            out = out.astype(np.uint8) * 255
        if out.dtype == np.double:
            out = out.astype(np.float32)
        tifffile.imwrite(output_name, out, imagej=True)


def has_metadata(im):
    """Check if an image has metadata."""
    return hasattr(im, 'metadata')


def to_stack(images):
    """Create a stack of images along the first axis."""
    if not images:
        return None
    img_list = []
    for img in images:
        if isinstance(img, str):
            img_list.append(Img(img))
        elif isinstance(img, Img):
            img_list.append(img)
        else:
            img_list.append(np.asarray(img))
    if not img_list:
        return None
    return np.stack(img_list, axis=0)


def create_compatible_image(img1, img2, auto_sort=False):
    """Make two images compatible in shape and dtype."""
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
    if img1.dtype != img2.dtype:
        if img1.dtype == np.uint8 or img2.dtype == np.uint8:
            img1 = img1.astype(np.uint8)
            img2 = img2.astype(np.uint8)
        else:
            img1 = img1.astype(float)
            img2 = img2.astype(float)
    return img1, img2


def blend(bg, fg, alpha=0.3, mask_or_forbidden_colors=None):
    """Blend foreground and background images with alpha transparency."""
    bg, fg = create_compatible_image(bg, fg, auto_sort=True)
    if bg.max() > 255 or bg.dtype != np.uint8:
        bg = np.interp(bg, (bg.min(), bg.max()), (0, 255))
    blended = fg * alpha + bg * (1. - alpha)
    blended = np.clip(blended, 0, 255)
    blended = blended.astype(np.uint8)
    if mask_or_forbidden_colors is None:
        return blended
    if not isinstance(mask_or_forbidden_colors, np.ndarray):
        mask_or_forbidden_colors = mask_colors(fg, colors_to_mask=mask_or_forbidden_colors)
    if mask_or_forbidden_colors is None:
        return blended
    bg, mask_or_forbidden_colors = create_compatible_image(bg, mask_or_forbidden_colors, auto_sort=False)
    if mask_or_forbidden_colors.dtype != bool:
        if mask_or_forbidden_colors.max() != 0:
            mask_or_forbidden_colors = mask_or_forbidden_colors / mask_or_forbidden_colors.max()
    bg = bg * mask_or_forbidden_colors
    bg = np.clip(bg, 0, 255)
    bg = bg.astype(np.uint8)
    blended[bg != 0] = bg[bg != 0]
    return blended


def mask_colors(colored_image, colors_to_mask, invert_mask=False, warn_on_color_not_found=False):
    """Create a mask based on specified colors in an image."""
    if colors_to_mask is None:
        logger.warning('No color to be masked was specified')
        return None
    if not (isinstance(colors_to_mask, list) or isinstance(colors_to_mask, tuple)):
        colors_to_mask = [colors_to_mask]
    # Initialize mask with correct shape: 2D for RGB images, same as image for grayscale
    if colored_image.ndim == 2:
        mask_shape = colored_image.shape
    else:
        # For RGB images, mask should be 2D (without channel dimension)
        mask_shape = colored_image.shape[:-1]
    if not invert_mask:
        mask = np.zeros(mask_shape, dtype=bool)
    else:
        mask = np.ones(mask_shape, dtype=bool)
    for color in colors_to_mask:
        if isinstance(color, (int, np.integer)):
            if colored_image.ndim == 2:
                mask |= (colored_image == color)
            else:
                rgb = int24_to_RGB(np.array([[color]], dtype=np.uint32))[0, 0]
                mask |= np.all(colored_image == rgb, axis=-1)
        elif isinstance(color, (tuple, list)) and len(color) == 3:
            mask |= np.all(colored_image == color, axis=-1)
    if invert_mask:
        mask = ~mask
    if not np.any(mask):
        if warn_on_color_not_found:
            logger.warning('No pixels found with specified colors')
        return None
    return mask


def pad_border_xy(img, dim_x=-2, dim_y=-3, size=1, mode='symmetric', **kwargs):
    """Pad the borders of an image."""
    if size <= 0:
        return img
    if img is None:
        logger.error('Image is None')
        return None
    if dim_y == -3 and abs(dim_y) > len(img.shape):
        if dim_x == -2:
            dim_x = -1
            dim_y = -2
    pad_seq = []
    for dim in range(len(img.shape)):
        if dim == (len(img.shape) + dim_x) % len(img.shape):
            pad_seq.append((size, size))
        elif dim == (len(img.shape) + dim_y) % len(img.shape):
            pad_seq.append((size, size))
        else:
            pad_seq.append((0, 0))
    return np.pad(img, pad_width=tuple(pad_seq), mode=mode, **kwargs)


def PIL_to_numpy(PIL_image):
    """Convert a PIL image to a numpy array."""
    return np.asarray(PIL_image)


def invert(img):
    """Invert the values of an image."""
    if not img.dtype == bool:
        mx = img.max()
        mn = img.min()
        img = np.negative(img) + mx + mn
    else:
        img = ~img
    return img


def _normalize_8bits(img, mode='min_max'):
    """Normalize image to 8-bit scale."""
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    if mode == 'min_max':
        mn = img.min()
        mx = img.max()
        if mx == mn:
            return img.astype(np.uint8)
        img = (img - mn) / (mx - mn) * 255
    return np.clip(img, 0, 255).astype(np.uint8)


def blend_stack_channels_color_mode(img, luts=None):
    """Blend stacked channels into RGB using LUTs."""
    if len(img.shape) == 2:
        return np.stack([img, img, img], axis=-1)
    final_image = np.zeros((*img.shape[:-1], 3), dtype=float)
    for ch in range(min(img.shape[-1], 3)):
        tmp = img[..., ch].astype(float)
        mn = tmp.min()
        mx = tmp.max()
        if mn != mx:
            tmp = (tmp - mn) / (mx - mn)
        elif mx != 0:
            tmp = tmp / mx
        final_image[..., ch] = tmp * 255
    return np.clip(final_image, 0, 255).astype(np.uint8)


def toQimage(img, autofix_always_display2D=True, normalize=True, z_behaviour=None, metadata=None, preserve_alpha=False):
    """Convert a numpy ndarray to a QImage."""
    from utils.qt_settings import set_UI
    set_UI()
    from qtpy.QtGui import QImage
    
    luts = None
    try:
        luts = img.metadata['LUTs']
    except:
        pass
    if metadata is not None:
        try:
            luts = metadata['LUTs']
        except:
            pass
    
    dimensions = None
    if isinstance(img, Img):
        try:
            dimensions = img.metadata.get('dimensions')
        except:
            pass
    
    if dimensions is None:
        if img.ndim == 2:
            dimensions = 'hw'
        elif img.ndim == 3:
            dimensions = 'hwc' if img.shape[2] <= 4 else 'dhwc'
        else:
            dimensions = 'hw'
    
    if autofix_always_display2D:
        if 'd' in dimensions and img.ndim >= 3:
            img = img[0]
        if 't' in dimensions and img.ndim >= 4:
            img = img[0]
    
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    
    nb_channels = img.shape[-1]
    
    if nb_channels == 3:
        if luts is not None:
            try:
                img = blend_stack_channels_color_mode(img, luts=luts)
            except:
                pass
        if normalize:
            img = _normalize_8bits(img)
        bytesPerLine = 3 * img.shape[-2]
        qimage = QImage(img.data.tobytes(), img.shape[-2], img.shape[-3], bytesPerLine, QImage.Format_RGB888)
    elif nb_channels < 3:
        try:
            bgra = blend_stack_channels_color_mode(img, luts=luts)
        except:
            bgra = np.zeros((img.shape[-3], img.shape[-2], 3), np.uint8, 'C')
            if img.shape[2] >= 1:
                bgra[..., 0] = img[..., 0]
            if img.shape[2] >= 2:
                bgra[..., 1] = img[..., 1]
            if img.shape[2] >= 3:
                bgra[..., 2] = img[..., 2]
        bytesPerLine = 3 * bgra.shape[-2]
        if normalize:
            bgra = _normalize_8bits(bgra)
        qimage = QImage(bgra.data.tobytes(), img.shape[-2], img.shape[-3], bytesPerLine, QImage.Format_RGB888)
    else:
        if nb_channels == 4 and preserve_alpha:
            bgra = np.zeros((img.shape[-3], img.shape[-2], 4), np.uint8, 'C')
            bgra[..., 0] = img[..., 0]
            bgra[..., 1] = img[..., 1]
            bgra[..., 2] = img[..., 2]
            if img.shape[2] >= 4:
                bgra[..., 3] = img[..., 3]
            else:
                bgra[..., 3].fill(255)
            bytesPerLine = 4 * bgra.shape[-2]
            qimage = QImage(bgra.data.tobytes(), img.shape[-2], img.shape[-3], bytesPerLine, QImage.Format_ARGB32)
        else:
            try:
                bgra = blend_stack_channels_color_mode(img, luts=luts)
            except:
                bgra = np.average(img, axis=-1)
                bgra = np.stack([bgra, bgra, bgra], axis=-1).astype(np.uint8)
            if normalize:
                bgra = _normalize_8bits(bgra)
            bytesPerLine = 3 * bgra.shape[-2]
            qimage = QImage(bgra.data.tobytes(), bgra.shape[-2], bgra.shape[-3], bytesPerLine, QImage.Format_RGB888)
    
    return qimage


__all__ = [
    'RGB_to_int24', 'int24_to_RGB', 'is_binary', 'auto_scale', 'get_white_bounds',
    'save_as_tiff', 'has_metadata', 'to_stack', 'blend', 'mask_colors',
    'pad_border_xy', 'PIL_to_numpy', 'invert', 'toQimage', '_create_dir'
]
