"""Lookup table (LUT) utilities for CellMap."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utils.image_utils import int24_to_RGB
from utils.logger import TA_logger

logger = TA_logger()


def cmap_to_numpy(lut):
    """Convert a colormap to a NumPy array."""
    if isinstance(lut, str):
        cm = plt.get_cmap(lut)
    else:
        cm = lut
    out = cm(range(256)) * 255
    out = out[..., 0:3].astype(np.uint8)
    return out


def numpy_to_cmap(lut):
    """Convert a NumPy array to a colormap."""
    lut_cp = np.copy(lut)
    if lut_cp.max() > 1:
        lut_cp = lut_cp / 255.
    if lut_cp.shape[-1] == 3:
        tmp = np.ones((256, 4), dtype=np.float64)
        tmp[..., 0:3] = lut_cp[..., :]
        lut_cp = tmp
    newcmp = ListedColormap(lut_cp)
    return newcmp


def apply_lut(img, lut, convert_to_RGB=True):
    """Apply a lookup table to an image."""
    if isinstance(lut, str):
        cmap = plt.get_cmap(lut)
    elif isinstance(lut, np.ndarray):
        cmap = numpy_to_cmap(lut)
    else:
        cmap = lut
    
    # Normalize image to 0-1 range
    img_norm = img.astype(float)
    if img_norm.max() > 1:
        img_norm = img_norm / 255.0
    
    # Apply colormap
    colored = cmap(img_norm)
    
    if convert_to_RGB:
        return (colored[..., :3] * 255).astype(np.uint8)
    return colored


class PaletteCreator:
    """Simple palette creator for LUTs."""
    
    def __init__(self):
        self.list = {
            'RED': 'red',
            'GREEN': 'green',
            'BLUE': 'blue',
            'GRAY': 'gray',
            'GREY': 'gray',
            'HOT': 'hot',
            'JET': 'jet',
            'VIRIDIS': 'viridis',
            'PLASMA': 'plasma',
            'INFERNO': 'inferno',
            'MAGMA': 'magma',
        }
    
    def create3(self, lut_name):
        """Create a 3-channel colormap."""
        if lut_name in self.list:
            return plt.get_cmap(self.list[lut_name])
        return plt.get_cmap('gray')


def list_available_luts(return_matplotlib_lib_luts_in_separate_database=False):
    """List available lookup tables."""
    hash_pal = {
        "DEFAULT": "gray",
        "GREEN": "green",
        "RED": "red",
        "BLUE": "blue",
        "GRAY": "gray",
        "GREY": "gray",
        "HOT": "hot",
        "JET": "jet",
        "VIRIDIS": "viridis",
        "PLASMA": "plasma",
        "INFERNO": "inferno",
        "MAGMA": "magma",
    }
    
    # Add matplotlib colormaps
    try:
        import matplotlib.cm as cm
        for name in dir(cm):
            if not name.startswith('_') and hasattr(getattr(cm, name), 'N'):
                hash_pal['MATPLOTLIB_' + name] = name
    except:
        pass
    
    if return_matplotlib_lib_luts_in_separate_database:
        matplot_lib_luts = [name.replace('MATPLOTLIB_', '') for name in hash_pal.keys() if name.startswith('MATPLOTLIB_')]
        return matplot_lib_luts, hash_pal
    
    return hash_pal


def matplotlib_to_TA(lut_name):
    """Convert matplotlib LUT name to TA format."""
    if lut_name.startswith('MATPLOTLIB_'):
        return lut_name.replace('MATPLOTLIB_', '')
    return lut_name


__all__ = ['apply_lut', 'PaletteCreator', 'list_available_luts', 'matplotlib_to_TA', 'cmap_to_numpy', 'numpy_to_cmap']
