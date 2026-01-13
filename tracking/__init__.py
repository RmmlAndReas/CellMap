"""Tracking module for cell tracking in tissue analysis.

This module provides the main tracking interface and organizes tracking methods
into separate modules for better maintainability.
"""

from tracking.core import (
    track_cells_dynamic_tissue,
    track_cells_static_tissue,
    match_by_max_overlap,
)
from tracking.utils.tools import (
    get_list_of_files,
    first_image_tracking,
    smart_name_parser,
    get_TA_file,
    get_mask_file,
    get_input_files_and_output_folders,
    assign_random_ID_to_missing_cells,
)
from tracking.utils.registration import (
    apply_translation,
    pre_register_images,
    get_pyramidal_registration,
)
from tracking.utils.track_correction import (
    swap_tracks,
    connect_tracks,
    correct_track,
)
from tracking.utils.local_to_track_correspondance import (
    add_localID_to_trackID_correspondance_in_DB,
    get_local_id_n_track_correspondence_from_images,
)

__all__ = [
    'track_cells_dynamic_tissue',
    'track_cells_static_tissue',
    'match_by_max_overlap',
    'get_list_of_files',
    'first_image_tracking',
    'smart_name_parser',
    'get_TA_file',
    'get_mask_file',
    'get_input_files_and_output_folders',
    'assign_random_ID_to_missing_cells',
    'apply_translation',
    'pre_register_images',
    'get_pyramidal_registration',
    'swap_tracks',
    'connect_tracks',
    'correct_track',
    'add_localID_to_trackID_correspondance_in_DB',
    'get_local_id_n_track_correspondence_from_images',
]
