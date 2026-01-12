"""List utility functions for file management and list operations."""

import os
import glob
import shutil
from natsort import natsorted
from utils.logger import TA_logger

logger = TA_logger()


def list_files_in_child_dirs(parent_directory, depth=1):
    """List files in child directories up to a specified depth."""
    tif_files = []
    parent_directory, ext = parent_directory.split('*')

    def recursive_list(directory, current_depth):
        if current_depth >= depth:
            return
        for child_dir in os.listdir(directory):
            child_dir_path = os.path.join(directory, child_dir)
            if os.path.isdir(child_dir_path):
                tif_files.extend(glob.glob(os.path.join(child_dir_path, f'*{ext}')))
                recursive_list(child_dir_path, current_depth + 1)

    recursive_list(parent_directory, current_depth=0)
    return tif_files


def create_list(input_folder, save=False, output_name=None, extensions=['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tga'], sort_type='natsort', recursive=False):
    """
    Create a list of files from a folder or pattern.
    
    Args:
        input_folder: Folder path or glob pattern
        save: Whether to save the list to a file
        output_name: Output file name (defaults to 'list.lst' in input_folder)
        extensions: List of file extensions to include
        sort_type: Sort type ('natsort' or None)
        recursive: Whether to search recursively (bool or int for depth)
    
    Returns:
        List of file paths
    """
    lst = []
    if not extensions:
        logger.error('Extensions are not defined --> list cannot be created')
        return lst
    path = input_folder
    if not path.endswith('/') and not path.endswith('\\') and not '*' in path and not os.path.isfile(path):
        path += '/'
    if not '*' in path:
        for ext in extensions:
            if not ext.startswith('*'):
                ext = '*' + ext
            lst += glob.glob(path + ext)
    else:
        if not recursive:
            lst += glob.glob(path)
        else:
            if isinstance(recursive, int):
                lst = list_files_in_child_dirs(path, depth=recursive)
            else:
                path, ext = path.split('*')
                for root, _, _ in os.walk(path):
                    lst.extend(glob.glob(os.path.join(root, '*'+ext)))
    if sort_type == 'natsort':
        lst = natsorted(lst)
    if save:
        if output_name is None:
            output_name = os.path.join(input_folder, 'list.lst')
        save_list_to_file(lst, output_name)
    return lst


def get_resume_list(input_list, resume_value):
    """Get a sublist starting from a resume value."""
    try:
        resume_index = input_list.index(resume_value)
        resume_list = input_list[resume_index:]
        return resume_list
    except ValueError:
        return []


def loadlist(txtfile, always_prefer_local_directory_if_file_exists=True, filter_existing_files_only=False,
             smart_local_folder_detection=True, skip_hashtag_commented_lines=True, skip_empty_lines=True, recursive=False):
    """
    Load a list of files from a text file or create from a pattern.
    
    Args:
        txtfile: Path to text file, directory, or glob pattern
        always_prefer_local_directory_if_file_exists: Prefer files in the list file's directory
        filter_existing_files_only: Only return files that exist
        smart_local_folder_detection: Use smart path detection
        skip_hashtag_commented_lines: Skip lines starting with #
        skip_empty_lines: Skip empty lines
        recursive: Whether to search recursively
    
    Returns:
        List of file paths or None if file doesn't exist
    """
    if not os.path.exists(txtfile) or not os.path.isfile(txtfile):
        if not '*' in txtfile and not os.path.isdir(txtfile):
            return None
        else:
            lst = create_list(txtfile, save=False, recursive=recursive)
    else:
        with open(txtfile) as f:
            if not skip_hashtag_commented_lines:
                lst = [line.rstrip() for line in f]
            else:
                lst = [line.rstrip() for line in f if not line.rstrip().startswith('#')]
        if skip_empty_lines:
            lst = [line.rstrip() for line in lst if not line.strip() == '']

    if always_prefer_local_directory_if_file_exists:
        logger.debug('Files located in the .lst/.txt file containing folder will be preferred if they exist.')
        list_file_directory = os.path.dirname(txtfile)
        if not smart_local_folder_detection:
            lst = [os.path.join(list_file_directory, os.path.basename(line)) if os.path.isfile(
                os.path.join(list_file_directory, os.path.basename(line))) else line for line in lst]
        else:
            common_path = os.path.commonprefix(lst)
            if not os.path.isdir(common_path):
                common_path = os.path.dirname(common_path)
            if common_path != '':
                relative_paths = [os.path.relpath(path, common_path) for path in lst]
                lst = [os.path.join(list_file_directory, rel) if os.path.isfile(
                    os.path.join(list_file_directory, rel)) else line for rel, line in zip(relative_paths, lst)]
    if filter_existing_files_only:
        logger.debug('Checking list for existing files')
        lst = [line for line in lst if os.path.isfile(line)]
    return lst


def TA_smart_pairing_list(TA_list, corresponding_list_of_files_to_inject_or_folder, stop_on_error=False):
    """Pair TA list with corresponding files."""
    if TA_list.lower().endswith('.txt') or TA_list.lower().endswith('.lst'):
        TA_list = loadlist(TA_list)
    if os.path.isdir(corresponding_list_of_files_to_inject_or_folder):
        matching_list = []
        for file in TA_list:
            filename0_without_path = os.path.basename(file)
            matching_list.append(os.path.join(corresponding_list_of_files_to_inject_or_folder, filename0_without_path))
    elif corresponding_list_of_files_to_inject_or_folder.lower().endswith('.txt') or \
            corresponding_list_of_files_to_inject_or_folder.lower().endswith('.lst'):
        matching_list = loadlist(corresponding_list_of_files_to_inject_or_folder)
    if len(matching_list) != len(TA_list):
        if stop_on_error:
            print('Input lists/folders don\'t match', len(TA_list), len(matching_list))
            return
        else:
            pass
    zipped = list(zip(matching_list, TA_list))
    return zipped


def smart_TA_list(main_list, name_to_search_first, alternative_name_if_name_to_search_first_does_not_exist=None):
    """
    Create a smart TA list by searching for files with a specific name pattern.
    
    Args:
        main_list: List of base file paths
        name_to_search_first: Name pattern to search for first
        alternative_name_if_name_to_search_first_does_not_exist: Alternative name if first not found
    
    Returns:
        List of file paths
    """
    lst = [os.path.join(os.path.splitext(line)[0], name_to_search_first) for line in main_list]
    if alternative_name_if_name_to_search_first_does_not_exist is not None:
        lst = [line if os.path.isfile(line) or not os.path.isfile(os.path.join(os.path.dirname(line), alternative_name_if_name_to_search_first_does_not_exist)) else os.path.join(os.path.dirname(line), alternative_name_if_name_to_search_first_does_not_exist) for line in lst]
    return lst


def save_list_to_file(lst, filename, col_separator='\t'):
    """
    Save a list to a text file.
    
    Args:
        lst: List to save (can contain strings or lists/tuples)
        filename: Output filename
        col_separator: Separator for multi-column entries
    
    Returns:
        The input list
    """
    if lst is not None:
        with open(filename, 'w') as f:
            for item in lst:
                if not isinstance(item, list) and not isinstance(item, tuple):
                    f.write("%s\n" % item)
                else:
                    f.write("%s\n" % col_separator.join([str(i) for i in item]))
        return lst
    else:
        raise ValueError('Empty list --> cannot save it to a file')


__all__ = ['loadlist', 'create_list', 'save_list_to_file', 'smart_TA_list', 'TA_smart_pairing_list', 'get_resume_list']
