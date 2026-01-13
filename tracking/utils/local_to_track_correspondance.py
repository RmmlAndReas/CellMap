from skimage.measure import regionprops, label
import numpy as np
from typing import Dict, Tuple, Optional
from tracking.utils.tools import smart_name_parser, get_mask_file
from utils.image_io import Img
from utils.image_utils import RGB_to_int24
from utils.early_stopper import early_stop
from utils.list_utils import loadlist
from database.sqlite_db import TAsql
import traceback
import os
from utils.logger import TA_logger # logging

logger = TA_logger()

def save_local_id_and_track_correspondence(local_id_track_correspondence, db_file, type='cell', centroids: Optional[Dict[int, Tuple[float, float]]] = None):
    """
    Saves the local ID and track correspondence in a database, along with centroids if provided.

    Args:
        local_id_track_correspondence (dict): The dictionary containing the local ID and track correspondence.
        db_file (str): The path to the database file.
        type (str, optional): The type of correspondence. Defaults to 'cell'.
        centroids (dict, optional): Dictionary mapping local_id to (center_x, center_y) tuple. Defaults to None.

    """
    # Open the database and save the file in it
    db = TAsql(filename_or_connection=db_file)
    table_name = type + '_tracks'
    db.drop_table(table_name)

    if local_id_track_correspondence is None:
        db.close()
        return

    # Change the dict to another dict
    local_ids = list(local_id_track_correspondence.keys())
    track_ids = list(local_id_track_correspondence.values())
    
    # Prepare data dictionary
    data_with_headers = {'local_id': local_ids, 'track_id': track_ids}
    
    # Add centroids if provided
    if centroids is not None:
        center_x = [centroids.get(local_id, (None, None))[0] for local_id in local_ids]
        center_y = [centroids.get(local_id, (None, None))[1] for local_id in local_ids]
        data_with_headers['center_x_cells'] = center_x
        data_with_headers['center_y_cells'] = center_y
    
    db.create_and_append_table(table_name, data_with_headers)
    db.close()


def get_local_id_n_track_correspondence_from_images(filename):
    """
    Retrieves the local ID and track correspondence from the images, along with centroids.

    Args:
        filename (str): The filename of the images.

    Returns:
        tuple: (dict, dict) containing:
            - dict: The dictionary containing the local ID and track correspondence (local_id -> track_id)
            - dict: The dictionary containing centroids (local_id -> (center_x, center_y))
            Returns None if an error occurs.

    """
    if filename is None:
        return None

    local_id_track_correspondence = {}
    centroids = {}

    try:
        tracked_image_path = smart_name_parser(filename, ordered_output='tracked_cells_resized.tif')

        cell_id_image = None

        # Try to get mask file (handCorrection.tif, outlines.tif, or handCorrection.png)
        filename_without_ext = smart_name_parser(filename, ordered_output='full_no_ext')
        mask_file_path = get_mask_file(filename_without_ext)
        
        # Also try handCorrection.png as fallback
        if not os.path.exists(mask_file_path):
            handCorrection1, _ = smart_name_parser(filename, ordered_output=['handCorrection.png', 'handCorrection.tif'])
            if os.path.isfile(handCorrection1):
                mask_file_path = handCorrection1
        
        if os.path.exists(mask_file_path):
            cell_id_image = Img(mask_file_path)
        else:
            logger.error(f'File not found (tried handCorrection.tif, outlines.tif, and handCorrection.png): {mask_file_path} please segment the images first')
            return None

        if len(cell_id_image.shape) >= 3:
            cell_id_image = cell_id_image[..., 0]
        cell_id_image = label(cell_id_image, connectivity=1, background=255)

        tracked_image = None
        if os.path.isfile(tracked_image_path):
            tracked_image = RGB_to_int24(Img(tracked_image_path))

        if tracked_image is None:
            logger.error('File not found ' + str(tracked_image_path) + ' please track cells first')
            return None

        for region in regionprops(cell_id_image):
            color = tracked_image[region.coords[0][0], region.coords[0][1]]
            if color == 0:
                logger.warning("Tracks and cells don't match, correspondence will be meaningless, please update your files")
                return None
            local_id_track_correspondence[region.label] = color
            # Extract centroid: regionprops returns (y, x), we need (x, y)
            centroid = region.centroid
            centroids[region.label] = (centroid[1], centroid[0])  # Swap to (x, y) order
    except:
        traceback.print_exc()
        logger.error('Something went wrong when converting local ID to tracks for file ' + str(filename))
        return None

    return (local_id_track_correspondence, centroids)

def add_localID_to_trackID_correspondance_in_DB(lst, progress_callback=None):
    """
    Adds the local ID to track ID correspondence to the database.

    Args:
        lst (list): The list of files.
        progress_callback (function, optional): The callback function for reporting progress. Defaults to None.

    """
    if lst is not None and lst:
        for lll, file in enumerate(lst):
            try:
                if early_stop.stop == True:
                    return
                if progress_callback is not None:
                    progress_callback.emit(int((lll / len(lst)) * 100))
                else:
                    print(str((lll / len(lst)) * 100) + '%')
            except:
                pass
            db_file = smart_name_parser(file, ordered_output='pyTA.db')
            result = get_local_id_n_track_correspondence_from_images(file)
            if result is None:
                continue
            local_to_global_correspondence, centroids = result
            save_local_id_and_track_correspondence(local_to_global_correspondence, db_file, centroids=centroids)


if __name__ == "__main__":

    if True:
        lst = loadlist('/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/list.lst')
        add_localID_to_trackID_correspondance_in_DB(lst)

    if False:
        # test_local_to_global_correspondece = {1: 0xFF0000, 2: 0x00FF00, 3: 0x0000FF}
        db_file_for_test = '/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series012/pyTA.db'

        result = get_local_id_n_track_correspondence_from_images('/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series012.png')
        if result is not None:
            test_local_to_global_correspondece, test_centroids = result
            # print(test_local_to_global_correspondece)
            save_local_id_and_track_correspondence(test_local_to_global_correspondece, db_file_for_test, centroids=test_centroids)
        # try get the stuff

        # pass

