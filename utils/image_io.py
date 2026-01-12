"""Image I/O utilities for CellMap.

This module provides the Img class for handling images with metadata support.
"""
import os
import json
import datetime as dt
import numpy as np
import warnings
from PIL import Image
import tifffile
from skimage import io as skio
from skimage.util import img_as_ubyte
from utils.logger import TA_logger

logger = TA_logger()

# Try to import optional dependencies
try:
    import czifile
    HAS_CZIFILE = True
except ImportError:
    HAS_CZIFILE = False
    logger.warning("czifile not available, .czi files will not be supported")

try:
    import read_lif
    HAS_READ_LIF = True
except ImportError:
    HAS_READ_LIF = False
    logger.warning("read_lif not available, .lif files will not be supported")


def _create_dir(filename):
    """Create directory if it doesn't exist."""
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


class Img(np.ndarray):
    """
    Image class extending numpy ndarray with metadata support.
    
    The Img class is a numpy ndarray with additional metadata support for
    image dimensions, voxel sizes, and other image properties.
    """
    
    def __new__(cls, *args, t=0, d=0, z=0, h=0, y=0, w=0, x=0, c=0, bits=8, 
                serie_to_open=None, dimensions=None, metadata=None, **kwargs):
        """
        Creates a new instance of the Img class.
        
        Parameters:
        -----------
        *args : array_like or str
            Input image data (numpy array, file path, or list of file paths)
        t, d, z, h, y, w, x, c : int
            Dimension sizes (time, depth, height, width, channels)
        bits : int
            Bits per pixel (8, 16, or 32)
        dimensions : str
            Dimension order string (e.g., 'hw', 'hwc', 'dhwc')
        metadata : dict
            Dictionary containing metadata entries
        """
        img = None
        
        # Initialize default metadata
        meta_data = {
            'dimensions': None,
            'bits': None,
            'vx': None,  # voxel x size
            'vy': None,  # voxel y size
            'vz': None,  # voxel z size
            'AR': None,  # aspect ratio
            'LUTs': None,
            'cur_d': 0,
            'cur_t': 0,
            'Overlays': None,
            'ROI': None,
            'timelapse': None,
            'creation_time': None,
        }
        
        # Update with provided metadata
        if metadata is not None:
            meta_data.update(metadata)
        elif len(args) > 0 and isinstance(args[0], Img):
            # Copy metadata from existing Img
            try:
                meta_data.update(args[0].metadata)
            except:
                pass
        
        if len(args) == 1:
            # Case 1: Input is already a numpy array
            if isinstance(args[0], np.ndarray):
                img = np.asarray(args[0]).view(cls)
                img.metadata = meta_data.copy()
                if dimensions is not None:
                    img.metadata['dimensions'] = dimensions
            
            # Case 2: Input is a file path or list of paths
            elif isinstance(args[0], (str, list)):
                logger.debug('loading ' + str(args[0]))
                
                if isinstance(args[0], str) and '*' not in args[0]:
                    # Single image file
                    meta, img_data = _read_image_file(args[0], serie_to_open=serie_to_open)
                    if img_data is None:
                        logger.error(f"Cannot create Img from missing file: {args[0]}")
                        return None
                    meta_data.update(meta)
                    meta_data['path'] = args[0]
                    if not meta_data.get('creation_time'):
                        meta_data['creation_time'] = str(dt.datetime.now())
                    img = np.asarray(img_data).view(cls)
                    img.metadata = meta_data
                else:
                    # Series of images
                    if isinstance(args[0], list):
                        image_list = args[0]
                    else:
                        import glob
                        from natsort import natsorted
                        image_list = natsorted(glob.glob(args[0]))
                    
                    # Read and stack images
                    images = []
                    for file in image_list:
                        _, img_data = _read_image_file(file)
                        images.append(img_data)
                    
                    if images:
                        img = np.stack(images, axis=0)
                        img = np.asarray(img).view(cls)
                        meta_data['path'] = args[0]
                        if not meta_data.get('creation_time'):
                            meta_data['creation_time'] = str(dt.datetime.now())
                        img.metadata = meta_data
        else:
            # Custom image creation from dimension specifications
            dims = []
            dim_names = []
            
            if t != 0:
                dim_names.append('t')
                dims.append(t)
            if z != 0 or d != 0:
                dim_names.append('d')
                dims.append(max(z, d))
            if h != 0 or y != 0:
                dim_names.append('h')
                dims.append(max(h, y))
            if w != 0 or x != 0:
                dim_names.append('w')
                dims.append(max(w, x))
            if c != 0:
                dim_names.append('c')
                dims.append(c)
            
            dimensions_str = ''.join(dim_names)
            meta_data['dimensions'] = dimensions_str
            
            # Determine dtype based on bits
            if bits == 16:
                dtype = np.uint16
            elif bits == 32:
                dtype = np.float32
            else:
                dtype = np.uint8
            
            meta_data['bits'] = bits
            if not meta_data.get('creation_time'):
                meta_data['creation_time'] = str(dt.datetime.now())
            
            img = np.asarray(np.zeros(tuple(dims), dtype=dtype)).view(cls)
            img.metadata = meta_data
        
        if img is None:
            logger.critical("Error, can't create image: invalid arguments, file not supported or file does not exist...")
            return None
        
        return img
    
    def __array_finalize__(self, obj):
        """Called when a new array is created from this one."""
        if obj is None:
            return
        self.metadata = getattr(obj, 'metadata', {
            'dimensions': None,
            'bits': None,
            'vx': None,
            'vy': None,
            'vz': None,
            'AR': None,
            'LUTs': None,
            'cur_d': 0,
            'cur_t': 0,
            'Overlays': None,
            'ROI': None,
            'timelapse': None,
            'creation_time': None,
        })
    
    def save(self, output_name, print_file_name=False, ijmetadata='copy', mode='IJ'):
        """
        Save the image to a file.
        
        Parameters:
        -----------
        output_name : str
            Output file path
        print_file_name : bool
            Whether to print the file name when saving
        ijmetadata : str
            Metadata handling mode ('copy' or None)
        mode : str
            Save mode ('IJ' for ImageJ compatibility or 'raw')
        """
        if print_file_name:
            print('saving', output_name)
        
        if output_name is None:
            logger.error("No output name specified... ignoring...")
            return
        
        _create_dir(output_name)
        
        # Handle TIFF files
        if output_name.lower().endswith('.tif') or output_name.lower().endswith('.tiff'):
            out = self.copy()
            
            # Type conversions for ImageJ compatibility
            if mode == 'IJ':
                if out.dtype == np.int32:
                    out = out.astype(np.float32)
                if out.dtype == np.int64:
                    out = out.astype(np.float64)
                if out.dtype == bool:
                    out = out.astype(np.uint8) * 255
                if out.dtype == np.double:
                    out = out.astype(np.float32)
                
                # Handle dimensions for ImageJ format
                if self.metadata.get('dimensions') is not None:
                    if not self.has_c():
                        out = out[..., np.newaxis]
                    if not self.has_d():
                        out = out[np.newaxis, ...]
                    if not self.has_t():
                        out = out[np.newaxis, ...]
                else:
                    if out.ndim < 3:
                        out = out[..., np.newaxis]
                    if out.ndim < 4:
                        out = out[np.newaxis, ...]
                    if out.ndim < 5:
                        out = out[np.newaxis, ...]
                
                # Move channel dimension for ImageJ order (TZCYX)
                out = np.moveaxis(out, -1, -3)
                
                # Prepare ImageJ metadata
                ijmeta = {}
                if ijmetadata == 'copy':
                    if self.metadata.get('Overlays'):
                        ijmeta['Overlays'] = self.metadata['Overlays']
                    if self.metadata.get('ROI'):
                        ijmeta['ROI'] = self.metadata['ROI']
                    if self.metadata.get('LUTs'):
                        ijmeta['LUTs'] = self.metadata['LUTs']
                
                try:
                    # Try new tifffile API
                    from tifffile.tifffile import imagej_metadata_tag
                    ijtags = imagej_metadata_tag(ijmeta, '>') if ijmeta else {}
                    tifffile.imwrite(
                        output_name, out, imagej=True,
                        metadata={'mode': 'composite'} if (
                            self.metadata.get('dimensions') and self.has_c()
                        ) else {},
                        extratags=ijtags
                    )
                except:
                    # Fallback to older API
                    tifffile.imwrite(
                        output_name, out, imagej=True,
                        metadata={'mode': 'composite'} if (
                            self.metadata.get('dimensions') and self.has_c()
                        ) else {}
                    )
            else:
                # Raw mode - just save as TIFF
                tifffile.imwrite(output_name, out)
        
        # Handle other formats (PNG, JPG, etc.)
        elif output_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            if not self.has_t() and not self.has_d():
                # Single 2D image
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    img_data = img_as_ubyte(self)
                new_im = Image.fromarray(img_data)
                new_im.save(output_name)
            else:
                logger.warning("Multi-dimensional image cannot be saved as single image file")
        
        # Handle numpy formats
        elif output_name.lower().endswith('.npy'):
            np.save(output_name, self, allow_pickle=False)
            # Save metadata separately
            if self.metadata:
                meta_copy = self.metadata.copy()
                if 'times' in meta_copy:
                    meta_copy['times'] = str(meta_copy['times'])
                with open(output_name + '.meta', 'w') as f:
                    json.dump(meta_copy, f)
        else:
            logger.error(f"Unsupported file format: {output_name}")
    
    def has_c(self):
        """Check if image has channel dimension."""
        dims = self.metadata.get('dimensions', '')
        if dims is None:
            dims = ''
        return 'c' in dims
    
    def has_d(self):
        """Check if image has depth dimension."""
        dims = self.metadata.get('dimensions', '')
        if dims is None:
            dims = ''
        return 'd' in dims
    
    def has_t(self):
        """Check if image has time dimension."""
        dims = self.metadata.get('dimensions', '')
        if dims is None:
            dims = ''
        return 't' in dims
    
    def get_dimension(self, dim):
        """Get the size of a specific dimension."""
        # Normalize dimension names
        if dim == 'z':
            dim = 'd'
        elif dim == 'x':
            dim = 'w'
        elif dim == 'y':
            dim = 'h'
        elif dim == 'f':
            dim = 't'
        
        dims = self.metadata.get('dimensions')
        if dims is None:
            logger.error(f'dimension {dim} not found!!!')
            return None
        
        if dim in dims:
            idx = dims.index(dim)
            idx = idx - len(dims)
            if self.ndim >= abs(idx) >= 1:
                return self.shape[idx]
            else:
                logger.error(f'dimension {dim} not found!!!')
                return None
        else:
            logger.error(f'dimension {dim} not found!!!')
            return None


def _read_image_file(filename, serie_to_open=None):
    """
    Read an image file and return metadata and image data.
    
    Returns:
    --------
    metadata : dict
        Image metadata
    image : ndarray
        Image data as numpy array
    """
    metadata = {
        'dimensions': None,
        'bits': None,
        'vx': None,
        'vy': None,
        'vz': None,
    }
    
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return metadata, None
    
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        if ext in ['.tif', '.tiff', '.lsm']:
            # Read TIFF file
            with tifffile.TiffFile(filename) as tif:
                # Extract ImageJ metadata if available
                if tif.is_imagej:
                    ij_meta = tif.imagej_metadata
                    if ij_meta:
                        if 'Overlays' in ij_meta:
                            metadata['Overlays'] = ij_meta['Overlays']
                        if 'ROI' in ij_meta:
                            metadata['ROI'] = ij_meta['ROI']
                        if 'LUTs' in ij_meta:
                            metadata['LUTs'] = ij_meta['LUTs']
                
                # Extract basic metadata from tags
                if tif.pages:
                    page = tif.pages[0]
                    if 'ImageWidth' in page.tags:
                        metadata['w'] = page.tags['ImageWidth'].value
                    if 'ImageLength' in page.tags:
                        metadata['h'] = page.tags['ImageLength'].value
                    if 'BitsPerSample' in page.tags:
                        bits = page.tags['BitsPerSample'].value
                        if isinstance(bits, tuple):
                            bits = bits[0]
                        metadata['bits'] = bits
            
            image = np.squeeze(tifffile.imread(filename))
            
        elif ext == '.czi' and HAS_CZIFILE:
            # Read CZI file
            with czifile.CziFile(filename) as czi:
                image = czi.asarray()
                image = np.squeeze(image)
                # Try to extract metadata
                try:
                    meta_data = czi.metadata(raw=False)
                    if 'ImageDocument' in meta_data:
                        info = meta_data['ImageDocument']['Metadata']['Information']['Image']
                        if 'SizeX' in info:
                            metadata['w'] = info['SizeX']
                        if 'SizeY' in info:
                            metadata['h'] = info['SizeY']
                        if 'SizeZ' in info:
                            metadata['d'] = info['SizeZ']
                        if 'SizeC' in info:
                            metadata['c'] = info['SizeC']
                        if 'ComponentBitCount' in info:
                            metadata['bits'] = info['ComponentBitCount']
                except:
                    pass
        
        elif ext == '.lif' and HAS_READ_LIF:
            # Read LIF file
            reader = read_lif.Reader(filename)
            series = reader.getSeries()
            if serie_to_open is None:
                chosen = series[0]
            else:
                if 0 <= serie_to_open < len(series):
                    chosen = series[serie_to_open]
                else:
                    logger.error('Series index out of range')
                    return metadata, None
            
            meta_data = chosen.getMetadata()
            metadata['vx'] = meta_data.get('voxel_size_x')
            metadata['vy'] = meta_data.get('voxel_size_y')
            metadata['vz'] = meta_data.get('voxel_size_z')
            metadata['w'] = meta_data.get('voxel_number_x')
            metadata['h'] = meta_data.get('voxel_number_y')
            metadata['d'] = meta_data.get('voxel_number_z')
            metadata['c'] = meta_data.get('channel_number')
            
            # Read image data
            t_frames = chosen.getNbFrames()
            channels = metadata['c']
            images = []
            for T in range(t_frames):
                zstack = None
                for i in range(channels):
                    cur_image = chosen.getFrame(T=T, channel=i)
                    if zstack is None:
                        zstack = cur_image
                    else:
                        zstack = np.stack([zstack, cur_image], axis=-1)
                if zstack is not None:
                    images.append(zstack)
            
            if images:
                image = np.stack(images, axis=0)
            else:
                image = None
        
        else:
            # Use skimage for other formats (PNG, JPG, etc.)
            image = skio.imread(filename)
            if image.ndim == 2:
                metadata['dimensions'] = 'hw'
            elif image.ndim == 3:
                if image.shape[2] <= 4:  # RGB/RGBA
                    metadata['dimensions'] = 'hwc'
                else:
                    metadata['dimensions'] = 'dhwc'
            metadata['bits'] = 8  # Assume 8-bit for standard formats
        
    except Exception as e:
        logger.error(f"Error reading image file {filename}: {e}")
        import traceback
        traceback.print_exc()
        return metadata, None
    
    # Infer dimensions if not set
    if metadata.get('dimensions') is None and image is not None:
        if image.ndim == 2:
            metadata['dimensions'] = 'hw'
        elif image.ndim == 3:
            if image.shape[2] <= 4:
                metadata['dimensions'] = 'hwc'
            else:
                metadata['dimensions'] = 'dhwc'
        elif image.ndim == 4:
            metadata['dimensions'] = 'dhwc'
        elif image.ndim == 5:
            metadata['dimensions'] = 'tdhwc'
    
    return metadata, image


__all__ = ['Img']
