"""Set of functions used to manage file I/O.

    Functions:
        create_folders : Checks if an input list of folders exists, and creates any that do not exist.
        collect_input_dfs : Collects raw NFL stats data from local files, and if insufficient, optionally pulls additional data from online source.
        collect_roster_filter : Loads a roster filter file, and returns whether the load was successful.
        move_logfile : Moves the logfile generated during program execution from a temporary location to a new folder.
"""  # fmt: skip

import logging
import os
import shutil

import pandas as pd

# Set up logger
logger = logging.getLogger("log")


def create_folders(folders):
    """Checks if an input list of folders exists, and creates any that do not exist.

        Recommended to execute this function before any save commands to ensure the save folder exists.

        Args:
            folders (list or str): Folder or list of folders that may or may not already exist.
                If a folder does exist, this function will not modify it. If a function does not exist,
                this function will create it.
                If a file is input, tries to get the parent folder.
    """  # fmt: skip

    # Handle case of single folder being passed
    if isinstance(folders, str):
        folders = [folders]

    for input_folder in folders:
        # Check if a file (with a file extension) is passed instead of a folder. Get parent folder if so.
        folder = "/".join(input_folder.split("/")[:-1]) + "/" if "." in input_folder else input_folder

        if not os.path.exists(folder):
            os.makedirs(folder)
            logger.info(f"Created folder {folder}")


def collect_input_dfs(years, weeks, local_file_paths, online_file_paths, online_avail=False):
    """Collects raw NFL stats data from local files, and if insufficient, optionally pulls additional data from online source.

        Args:
            years (list or int): year or list of years to load raw input data from
            weeks (list): weeks within each year to load raw input data from
            local_file_paths (dict): dictionary with each type of input (e.g. 'pbp') as keys and filepaths to each type of input as values
            online_file_paths (dict): dictionary with each type of input (e.g. 'pbp') as keys and filepaths to each type of input as values.
                Keys must match between local_file_paths and online_file_paths.
            online_avail (bool, optional): toggle whether to allow pulling additional data from online files as necessary. Defaults to False.

        Returns:
            list | tuple: if a single year is input, returns a tuple of DataFrame objects corresponding to each input file type (e.g. 'pbp','roster').
                if multiple years are input, returns a list of tuples of DataFrame objects where each tuple corresponds to a year.
    """  # fmt: skip

    # Handle single year being input
    if not hasattr(years, "__iter__"):
        years = [years]

    # Create local folders if they do not exist
    local_folders = ["/".join(val.split("/")[:-1]) + "/" for val in local_file_paths.values()]
    create_folders(local_folders)

    # Load files one year at a time
    all_dfs = []
    for year in years:
        # Load local files
        try:
            yearly_dfs = tuple(pd.read_csv(local_file_paths[name].format(year), low_memory=False) for name in local_file_paths)
            logger.info("\n".join([f"Read {name} from {local_file_paths[name].format(year)}" for name in local_file_paths]))
            # Check if local files contain all weeks (checks all df's together)
            weeks_present = [all(any(year_df["week"] == week) for year_df in yearly_dfs) for week in weeks]
        except FileNotFoundError:
            weeks_present = [False]

        # Download files from online and save locally (updates all df's together)
        if not all(weeks_present):
            logger.info(f"Missing weeks in local data files for {year}.")
            yearly_dfs = ()
            if online_avail:
                for name in local_file_paths:
                    # Read from online filepath
                    logger.info(f"Downloading {name} from {online_file_paths[name].format(year)}")
                    year_online_df = pd.read_csv(online_file_paths[name].format(year), low_memory=False)
                    yearly_dfs = (*yearly_dfs, year_online_df)
                    # Save locally
                    year_online_df.to_csv(local_file_paths[name].format(year))
                    logger.info(f"Saved {name} to {local_file_paths[name].format(year)}")
            else:
                logger.warning("Warning! Not all weeks are present in the dfs, and could not download from online")

        all_dfs.append(yearly_dfs)

    # If only one year is input, no need to output a list of tuples, just the one tuple
    if len(years) == 1:
        return all_dfs[0]

    return all_dfs


def collect_roster_filter(filter_roster, update_filter, roster_filter_file):
    """Loads a roster filter file, and returns whether the load was successful.

        Args:
            filter_roster (bool): whether to filter roster at all (if False, no point in loading a filter file)
            update_filter (bool): whether to force an update of the roster filter (if True, no point in loading an old filter file)
            roster_filter_file (str): filepath to the roster filter file to use

        Returns:
            [pandas.DataFrame | None]: If file was loaded successfully, returns a DataFrame with the roster filter (player list). If file was
                not loaded successfully (either failed to load, or was not attempted), returns None.
            bool: True if file load was successful, False if not.
    """  # fmt: skip

    if filter_roster and not update_filter:
        try:
            filter_df = pd.read_csv(roster_filter_file)
            load_success = True
        except FileNotFoundError:
            filter_df = None
            load_success = False
    else:
        filter_df = None
        load_success = False

    return filter_df, load_success


def move_logfile(curr_filepath, new_folder):
    """Moves the logfile generated during program execution from a temporary location to a new folder.

        Args:
            curr_filepath (str): filepath where the logfile has been temporarily stored (MUST INCLUDE FILE NAME)
            new_folder (str): folder to move the logfile to (MUST NOT INCLUDE FILE NAME)
    """  # fmt: skip

    # Create folder if it does not exist
    create_folders(new_folder)

    logger.info(f"Saving logfile to {new_folder}")

    # Move file to new path
    filename = curr_filepath.split("/")[-1]
    new_filepath = new_folder + filename
    shutil.move(curr_filepath, new_filepath)
