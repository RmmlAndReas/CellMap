# zut le code est pas finalisé...

# USE a track object to store the click and its content...


# do a code to do smart track correction that avoids creating dupes
# checks for dupes, etc...
# can connect tracks and or swap tracks

# si les deux couleurs existent dans les deux images alors c'est du swap et faut swapper à partir de la dernière image sinon
# si chaque couleur existe dans une image et pas dans l'autre alors faire un connect track. Si le connect risque de dupliquer une cellule alors il faut faire un swap de la cellule puis connecter la track afin de ne pas dupliquer les cellules
# reflechir mais ça devrait etre bon
# existe t'il un moyen perenne de restaurer les tracks corrected ???? si oui lequel pr que le job ne soit pas perdu
# peut etre simplement faire un truc du genre update tracks et rajouter les cellules perdues de maniere aleatoire ? pas si simple en fait
# puis je aussi swapper plusieurs cellules en même temps ??? peut etre mais pas si simple
# comment faire un algo de swap intelligent


# need to store cell id or centroid or centre of area and the frame --> should be sufficient to do everything --> the most robust then detect the color of the other

# offer apply to all or just before or after the current image
# do the codes for that

# aussi faire un code pour reappliquer les tracks


from utils.image_utils import RGB_to_int24, int24_to_RGB
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
from utils.image_io import Img
from utils.list_utils import loadlist
from tracking.utils import tools
from utils.logger import TA_logger  # logging

logger = TA_logger()


def swap(img, id1, id2):
    """
    Swaps the pixel values of two IDs in the image.

    Args:
        img (numpy.ndarray): The input image.
        id1: The first ID to swap.
        id2: The second ID to swap.

    Returns:
        numpy.ndarray: The modified image with swapped IDs.

    """
    idcs1 = img == id1
    idcs2 = img == id2
    img[idcs1], img[idcs2] = id2, id1
    return img


def assign_id(img, old_id, new_id):
    """
    Assigns a new ID to all pixels with the specified old ID in the image.

    Args:
        img (numpy.ndarray): The input image.
        old_id: The ID to be replaced.
        new_id: The new ID to assign.

    Returns:
        numpy.ndarray: The modified image with updated IDs.

    """
    img[img == old_id] = new_id
    return img


def swap_tracks(lst, frame_of_first_connection, id_t0, id_t1, __preview_only=False):
    """
    Swaps the IDs of two tracks in a list of tracks.

    Args:
        lst (list): The list of tracks.
        frame_of_first_connection: The frame at which the first connection occurs.
        id_t0: The ID of the first track.
        id_t1: The ID of the second track.
        __preview_only (bool, optional): Whether to only preview the swap. Defaults to False.

    """
    correct_track(lst, frame_of_first_connection, id_t0, id_t1, correction_mode='swap', __preview_only=__preview_only)


def connect_tracks(lst, frame_of_first_connection, id_t0, id_t1, __preview_only=False):
    """
    Connects two tracks in a list of tracks.
    For merge operations: merges id_t1 (track B) into id_t0 (track A).
    
    Merge logic:
    - Track A (id_t0) before merge point: Keep as id_t0 (preceding frames kept)
    - Track A (id_t0) at merge point: Change to id_t1 (becomes track B)
    - Track B (id_t1) before merge point: Keep as id_t1 (if exists)
    - Track B (id_t1) at merge point and onwards: Already id_t1 (merged track continues)
    
    If tracks had predecessors (different IDs in previous frames), give them new IDs.

    Args:
        lst (list): The list of tracks.
        frame_of_first_connection: The frame at which the first connection occurs (merge point).
        id_t0: The ID of the first track (track A).
        id_t1: The ID of the second track (track B).
        __preview_only (bool, optional): Whether to only preview the connection. Defaults to False.

    """
    if frame_of_first_connection > 0:
        # Get all existing track IDs to avoid collisions when generating new IDs
        existing_ids = set()
        for l in range(len(lst)):
            tracked_cell_path = tools.smart_name_parser(lst[l], ordered_output='tracked_cells_resized.tif')
            try:
                tracked_cells = RGB_to_int24(Img(tracked_cell_path))
                existing_ids.update(np.unique(tracked_cells))
            except:
                pass
        
        from colors.colorgen import get_unique_random_color_int24
        
        # Detect preceding tracks (tracks that become id_t0 or id_t1 in frame N-1)
        preceding_id_t0 = None
        preceding_id_t1 = None
        if frame_of_first_connection > 0:
            try:
                prev_frame_path = tools.smart_name_parser(lst[frame_of_first_connection - 1], ordered_output='tracked_cells_resized.tif')
                prev_tracked_cells = RGB_to_int24(Img(prev_frame_path))
                
                # Find what track IDs were at the same cell positions in previous frame
                # For track A: find what ID was at the position where id_t0 appears in merge frame
                merge_frame_path = tools.smart_name_parser(lst[frame_of_first_connection], ordered_output='tracked_cells_resized.tif')
                merge_tracked_cells = RGB_to_int24(Img(merge_frame_path))
                
                # Get positions where id_t0 and id_t1 appear in merge frame
                id_t0_positions = np.where(merge_tracked_cells == id_t0)
                id_t1_positions = np.where(merge_tracked_cells == id_t1)
                
                if len(id_t0_positions[0]) > 0:
                    # Check what ID was at id_t0's position in previous frame
                    first_pos_idx = 0
                    prev_id_at_t0_pos = prev_tracked_cells[id_t0_positions[0][first_pos_idx], id_t0_positions[1][first_pos_idx]]
                    if prev_id_at_t0_pos != id_t0 and prev_id_at_t0_pos != 0xFFFFFF:
                        preceding_id_t0 = prev_id_at_t0_pos
                        logger.info(f'Found preceding track for track A: {preceding_id_t0:06x} -> {id_t0:06x}')
                
                if len(id_t1_positions[0]) > 0:
                    # Check what ID was at id_t1's position in previous frame
                    first_pos_idx = 0
                    prev_id_at_t1_pos = prev_tracked_cells[id_t1_positions[0][first_pos_idx], id_t1_positions[1][first_pos_idx]]
                    if prev_id_at_t1_pos != id_t1 and prev_id_at_t1_pos != 0xFFFFFF:
                        preceding_id_t1 = prev_id_at_t1_pos
                        logger.info(f'Found preceding track for track B: {preceding_id_t1:06x} -> {id_t1:06x}')
            except Exception as e:
                logger.warning(f'Error detecting preceding tracks: {e}')
        
        # Generate new IDs for preceding tracks if they exist
        new_id_for_preceding_t0 = None
        new_id_for_preceding_t1 = None
        
        if preceding_id_t0 is not None:
            new_id_for_preceding_t0 = get_unique_random_color_int24(forbidden_colors=list(existing_ids))
            existing_ids.add(new_id_for_preceding_t0)
            logger.info(f'Assigning new ID {new_id_for_preceding_t0:06x} to preceding track {preceding_id_t0:06x} of track A')
        
        if preceding_id_t1 is not None:
            new_id_for_preceding_t1 = get_unique_random_color_int24(forbidden_colors=list(existing_ids))
            existing_ids.add(new_id_for_preceding_t1)
            logger.info(f'Assigning new ID {new_id_for_preceding_t1:06x} to preceding track {preceding_id_t1:06x} of track B')
        
        # Process frames before merge point
        for l in range(frame_of_first_connection):
            tracked_cell_path = tools.smart_name_parser(lst[l], ordered_output='tracked_cells_resized.tif')
            try:
                tracked_cells = RGB_to_int24(Img(tracked_cell_path))
                modified = False
                
                # Give preceding tracks new IDs
                if preceding_id_t0 is not None and new_id_for_preceding_t0 is not None:
                    if preceding_id_t0 in tracked_cells:
                        tracked_cells = assign_id(tracked_cells, preceding_id_t0, new_id_for_preceding_t0)
                        logger.info(f'Frame {l}: Changed preceding track {preceding_id_t0:06x} to new ID {new_id_for_preceding_t0:06x}')
                        modified = True
                
                if preceding_id_t1 is not None and new_id_for_preceding_t1 is not None:
                    if preceding_id_t1 in tracked_cells:
                        tracked_cells = assign_id(tracked_cells, preceding_id_t1, new_id_for_preceding_t1)
                        logger.info(f'Frame {l}: Changed preceding track {preceding_id_t1:06x} to new ID {new_id_for_preceding_t1:06x}')
                        modified = True
                
                # Track A (id_t0) before merge: Keep as id_t0 (no change needed)
                # Track B (id_t1) before merge: Keep as id_t1 (no change needed)
                
                if modified and not __preview_only:
                    try:
                        import os
                        file_mtime_before = None
                        if os.path.exists(tracked_cell_path):
                            file_mtime_before = os.path.getmtime(tracked_cell_path)
                        
                        logger.info(f'Saving tracked cells to {tracked_cell_path} (frame {l}, before merge)')
                        Img(int24_to_RGB(tracked_cells), dimensions='hwc').save(tracked_cell_path, mode='raw')
                        
                        if os.path.exists(tracked_cell_path):
                            file_mtime_after = os.path.getmtime(tracked_cell_path)
                            if file_mtime_before != file_mtime_after:
                                logger.info(f'Successfully saved tracked cells to {tracked_cell_path} (mtime changed)')
                            else:
                                logger.warning(f'File {tracked_cell_path} exists but mtime unchanged')
                    except Exception as e:
                        logger.error(f'Failed to save tracked cells to {tracked_cell_path}: {e}')
                        import traceback
                        traceback.print_exc()
                        raise
            except Exception as e:
                logger.warning(f'Error processing frame {l} before merge: {e}')
                continue
        
        # At merge point: Change track A (id_t0) to track B (id_t1)
        try:
            merge_frame_path = tools.smart_name_parser(lst[frame_of_first_connection], ordered_output='tracked_cells_resized.tif')
            tracked_cells = RGB_to_int24(Img(merge_frame_path))
            modified = False
            
            if id_t0 in tracked_cells:
                tracked_cells = assign_id(tracked_cells, id_t0, id_t1)
                logger.info(f'Frame {frame_of_first_connection}: Changed track A ({id_t0:06x}) to track B ({id_t1:06x}) at merge point')
                modified = True
            
            if modified and not __preview_only:
                try:
                    import os
                    file_mtime_before = None
                    if os.path.exists(merge_frame_path):
                        file_mtime_before = os.path.getmtime(merge_frame_path)
                    
                    logger.info(f'Saving tracked cells to {merge_frame_path} (frame {frame_of_first_connection}, merge point)')
                    Img(int24_to_RGB(tracked_cells), dimensions='hwc').save(merge_frame_path, mode='raw')
                    
                    if os.path.exists(merge_frame_path):
                        file_mtime_after = os.path.getmtime(merge_frame_path)
                        if file_mtime_before != file_mtime_after:
                            logger.info(f'Successfully saved tracked cells to {merge_frame_path} (mtime changed)')
                        else:
                            logger.warning(f'File {merge_frame_path} exists but mtime unchanged')
                except Exception as e:
                    logger.error(f'Failed to save tracked cells to {merge_frame_path}: {e}')
                    import traceback
                    traceback.print_exc()
                    raise
        except Exception as e:
            logger.warning(f'Error processing merge point frame: {e}')
    
    # Process frames after merge point: Check if track A (id_t0) continues after merge point
    # If it does, give it a new ID (since track A became track B at merge point)
    if frame_of_first_connection + 1 < len(lst):
        # Check if id_t0 exists in frames after merge point
        continuing_track_a_exists = False
        for l in range(frame_of_first_connection + 1, len(lst)):
            tracked_cell_path = tools.smart_name_parser(lst[l], ordered_output='tracked_cells_resized.tif')
            try:
                tracked_cells = RGB_to_int24(Img(tracked_cell_path))
                if id_t0 in tracked_cells:
                    continuing_track_a_exists = True
                    break
            except:
                pass
        
        if continuing_track_a_exists:
            # Get all existing track IDs to avoid collisions
            existing_ids = set()
            for l in range(len(lst)):
                tracked_cell_path = tools.smart_name_parser(lst[l], ordered_output='tracked_cells_resized.tif')
                try:
                    tracked_cells = RGB_to_int24(Img(tracked_cell_path))
                    existing_ids.update(np.unique(tracked_cells))
                except:
                    pass
            
            from colors.colorgen import get_unique_random_color_int24
            new_id_for_continuing_track_a = get_unique_random_color_int24(forbidden_colors=list(existing_ids))
            logger.info(f'Assigning new ID {new_id_for_continuing_track_a:06x} to continuing track {id_t0:06x} (track A after merge point)')
            
            # Process frames after merge point: change continuing track A to new ID
            for l in range(frame_of_first_connection + 1, len(lst)):
                tracked_cell_path = tools.smart_name_parser(lst[l], ordered_output='tracked_cells_resized.tif')
                try:
                    tracked_cells = RGB_to_int24(Img(tracked_cell_path))
                    modified = False
                    
                    if id_t0 in tracked_cells:
                        tracked_cells = assign_id(tracked_cells, id_t0, new_id_for_continuing_track_a)
                        logger.info(f'Frame {l}: Changed continuing track A ({id_t0:06x}) to new ID {new_id_for_continuing_track_a:06x}')
                        modified = True
                    
                    if modified and not __preview_only:
                        try:
                            import os
                            file_mtime_before = None
                            if os.path.exists(tracked_cell_path):
                                file_mtime_before = os.path.getmtime(tracked_cell_path)
                            
                            logger.info(f'Saving tracked cells to {tracked_cell_path} (frame {l}, after merge)')
                            Img(int24_to_RGB(tracked_cells), dimensions='hwc').save(tracked_cell_path, mode='raw')
                            
                            if os.path.exists(tracked_cell_path):
                                file_mtime_after = os.path.getmtime(tracked_cell_path)
                                if file_mtime_before != file_mtime_after:
                                    logger.info(f'Successfully saved tracked cells to {tracked_cell_path} (mtime changed)')
                                else:
                                    logger.warning(f'File {tracked_cell_path} exists but mtime unchanged')
                        except Exception as e:
                            logger.error(f'Failed to save tracked cells to {tracked_cell_path}: {e}')
                            import traceback
                            traceback.print_exc()
                            raise
                except Exception as e:
                    logger.warning(f'Error processing frame {l} after merge: {e}')
                    continue

def correct_track(lst, frame_of_first_connection, id_t0, id_t1, correction_mode='connect', __preview_only=False, early_stop=True):
    """
    Corrects a track in a list of tracks based on the specified correction mode.

    Args:
        lst (list): The list of tracks.
        frame_of_first_connection: The frame at which the first connection occurs.
        id_t0: The ID of the first track.
        id_t1: The ID of the second track.
        correction_mode (str, optional): The correction mode. Defaults to 'connect'.
        __preview_only (bool, optional): Whether to only preview the correction. Defaults to False.
        early_stop (bool, optional): Whether to stop the correction process early. Defaults to True.

    """
    if frame_of_first_connection >= len(lst):
        logger.error('error wrong connection frame nb')
        return

    for l in range(frame_of_first_connection, len(lst)):
        tracked_cell_path = tools.smart_name_parser(lst[l], ordered_output='tracked_cells_resized.tif')
        tracked_cells_t0 = RGB_to_int24(Img(tracked_cell_path))

        cellpos_1 = id_t0 in tracked_cells_t0
        cellpos_2 = id_t1 in tracked_cells_t0

        if not cellpos_1 and not cellpos_2:
            logger.info('cells not found --> ignoring track correction')
            continue

        # For connect mode (merge), we allow merging even if both tracks exist in the same frame
        # This is intentional - we're merging id_t1 into id_t0, so both existing is fine
        # The check below was incorrectly blocking merges when both tracks coexist
        # (This check was meant to prevent accidental duplication, but merge is intentional)

        # Only check for missing cells in swap mode
        if correction_mode == 'swap' and ((not cellpos_1 and cellpos_2) or (not cellpos_2 and cellpos_1)):
            logger.info('missing cell at frame ' + str(l) + ' swapping ignored')
            if early_stop:
                logger.info('quitting track edition')
                break
            else:
                continue

        if cellpos_1 and cellpos_2 and correction_mode == 'swap':
            logger.info('swapped cell ids: ' + str(id_t0) + ' and ' + str(id_t1))
            tracked_cells_t0 = swap(tracked_cells_t0, id_t0, id_t1)
            if __preview_only:
                plt.imshow(int24_to_RGB(tracked_cells_t0))
                plt.show()
            else:
                try:
                    import os
                    # Get file modification time before save
                    file_mtime_before = None
                    if os.path.exists(tracked_cell_path):
                        file_mtime_before = os.path.getmtime(tracked_cell_path)
                    
                    logger.info(f'Saving tracked cells to {tracked_cell_path} (frame {l}, swap)')
                    Img(int24_to_RGB(tracked_cells_t0), dimensions='hwc').save(tracked_cell_path, mode='raw')
                    
                    # Verify the file was actually written
                    if os.path.exists(tracked_cell_path):
                        file_mtime_after = os.path.getmtime(tracked_cell_path)
                        if file_mtime_before != file_mtime_after:
                            logger.info(f'Successfully saved tracked cells to {tracked_cell_path} (mtime changed)')
                        else:
                            logger.warning(f'File {tracked_cell_path} exists but mtime unchanged - save may have failed')
                    else:
                        logger.error(f'File {tracked_cell_path} does not exist after save operation!')
                except Exception as e:
                    logger.error(f'Failed to save tracked cells to {tracked_cell_path}: {e}')
                    import traceback
                    traceback.print_exc()
                    raise  # Re-raise to ensure the error is visible
        else:
            # For connect mode: only merge id_t1 into id_t0 (keep id_t0, change id_t1 to id_t0)
            # Do NOT change id_t0 to id_t1 - we want to keep the first track's ID
            if cellpos_2:
                # Track id_t1 is present - merge it into id_t0
                logger.info('changed id ' + str(id_t1) + ' to ' + str(id_t0))
                tracked_cells_t0 = assign_id(tracked_cells_t0, id_t1, id_t0)
            # If only cellpos_1 (id_t0) is present, do nothing - keep it as is
            if __preview_only:
                plt.imshow(int24_to_RGB(tracked_cells_t0))
                plt.show()
            else:
                try:
                    import os
                    # Get file modification time before save
                    file_mtime_before = None
                    if os.path.exists(tracked_cell_path):
                        file_mtime_before = os.path.getmtime(tracked_cell_path)
                    
                    logger.info(f'Saving tracked cells to {tracked_cell_path} (frame {l})')
                    Img(int24_to_RGB(tracked_cells_t0), dimensions='hwc').save(tracked_cell_path, mode='raw')
                    
                    # Verify the file was actually written
                    if os.path.exists(tracked_cell_path):
                        file_mtime_after = os.path.getmtime(tracked_cell_path)
                        if file_mtime_before != file_mtime_after:
                            logger.info(f'Successfully saved tracked cells to {tracked_cell_path} (mtime changed)')
                        else:
                            logger.warning(f'File {tracked_cell_path} exists but mtime unchanged - save may have failed')
                    else:
                        logger.error(f'File {tracked_cell_path} does not exist after save operation!')
                except Exception as e:
                    logger.error(f'Failed to save tracked cells to {tracked_cell_path}: {e}')
                    import traceback
                    traceback.print_exc()
                    raise  # Re-raise to ensure the error is visible

if __name__ == "__main__":
    start_all = timer()

    # path = '/E/Sample_images/tracking_test'
    path = '/E/Sample_images/sample_images_PA/mini10_fake_swaps/liste.lst'
    # path = '/E/Sample_images/test_tracking_with_shift' # quick test of the algo
    #
    # images_to_analyze = os.listdir(path)
    # images_to_analyze = [os.path.join(path, f) for f in images_to_analyze if
    #                      os.path.isfile(os.path.join(path, f))]  # list only files and only if they end by tif
    # images_to_analyze = natsorted(images_to_analyze)

    images_to_analyze = loadlist(path)

    # then do amrutha's code watershed  imagendarray (2-D, 3-D, …) of integers--> cool --> will also work for 3D --> then need segmentation in 3D
    #
    #
    # for l in range(len(images_to_analyze)):
    #     # in fact I don't even need to reopen the mask file because the track file should suffice ???
    #
    #     file_path_0 = images_to_analyze[l]
    #     if not file_path_0.endswith('.tif'):
    #         continue
    #
    #
    #     # print('files', file_path_1, file_path_0)
    #
    #     filename0_without_ext = os.path.splitext(file_path_0)[0]
    #
    #     tracked_cells_t0 = Img(os.path.join(filename0_without_ext,        'tracked_cells_resized.tif')) # 455:455 + 128, 435:435 + 128 #455:455 + 256, 435:435 + 290 #768:1024,0:256 #0:128, 0:128 #500:500+128, 500:500+128
    #
    #
    #
    #
    #     # TODO always get the mask instead of any other files in order to have the minimal nb of files available
    #     # mask_t0 = Img(os.path.join(filename0_without_ext, 'handCorrection.tif'))[
    #     #     ..., 0]  # 455:455 + 128, 435:435 + 128 #455:455 + 256, 435:435 + 290 #768:1024,0:256 #0:128, 0:128 #500:500+128, 500:500+128
    #
    #
    #     # labels_t0 = measure.label(mask_t0, connectivity=1,
    #     #                           background=255)  # FOUR_CONNECTED # use bg = 255 to avoid having to invert the image --> a gain of time I think and could be applied more generally
    #
    #     height = tracked_cells_t0.shape[0]
    #     width = tracked_cells_t0.shape[1]
    #
    #
    #     # could maybe store the ID of duplicate cell as a reseed --> can be a pb with extruding cells but should most often be ok though
    #     #
    #
    #     labels_tracking_t0 = measure.label(tracked_cells_t0[..., 0], connectivity=1, background=255)
    #     rps_t0 = regionprops(labels_tracking_t0)
    #
    #
    #     centroids_t0 = []
    #     for region in rps_t0:
    #         # take regions with large enough areas
    #         centroids_t0.append(region.centroid)

    # try create a dupe to see if it fails -> it will and I should really avoid that

    # swap_tracks(images_to_analyze, 1, 0x0ef5dd, 0xa28ad6, __preview_only=True)  # both cells exist


    swap_tracks(images_to_analyze, 0, 0x0ef5dd, 0xFAFAFF, __preview_only=True)  # one cells exist

    # ça marche mais reflechir que je ne puisse pas creer des pbs avec ça
    # connect_tracks(images_to_analyze, 0, 0xffdd5d, 0x3b2cdb, __preview_only=True)  # both cells exist
    # connect_tracks(images_to_analyze, 0, 0xffdd5d, 0x0ef5dd, __preview_only=True)  # create a duplicate cell # --> breaking

    # codes are the same --> just keep one and

    # tt a l'air de marcher --> maybe add a flag for recursive --> in fact not...

    print('total time', timer() - start_all)

