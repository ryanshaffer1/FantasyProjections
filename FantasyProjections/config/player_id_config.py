"""Contains variables defining the Player ID systems used throughout the program, and functions to work with/manipulate these Player ID systems.

    Variables:
        PRIMARY_PLAYER_ID (str): Name of the system used as primary player ID (should be complete and unique across all NFL players). Example: "gsis_id"
        ALT_PLAYER_IDS (list): Names of alternate player IDs used when interacting with other data sources. Example: "pfr_id" (pro-football-reference.com system).
        PLAYER_IDS (list): Combines the PRIMARY_PLAYER_ID and ALT_PLAYER_IDS into one list of all IDs that should be tracked for each player.

    Functions:
        fill_blank_player_ids : Attempts to add missing alternate player IDs to a DataFrame of players, using a master list and other resources to search for the correct player IDs.
        update_master_player_ids : Adds potentially new players to the master Player IDs file, and optionally searches for matches to missing IDs in the file.
"""  # fmt: skip

import json
import logging
import logging.config

import pandas as pd
from config.log_config import LOGGING_CONFIG
from data_pipeline.stats_pipeline.scrape_pro_football_reference import search_for_missing_pfr_id
from data_pipeline.utils.name_matching import find_matching_name_ind

# Define constants for ID systems being used and tracked throughout the program
PRIMARY_PLAYER_ID = "gsis_id"
ALT_PLAYER_IDS = ["pfr_id", "sleeper_id"]
PLAYER_IDS = [PRIMARY_PLAYER_ID, *ALT_PLAYER_IDS]


# Functions to manipulate Player ID systems

# Set up logger
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")


def fill_blank_player_ids(players_df, master_id_file=None, pfr_id_filename=None, *, add_missing_pfr=False, update_master=False):
    """Attempts to add missing alternate player IDs to a DataFrame of players, using a master list and other resources to search for the correct player IDs.

        Args:
            players_df (pandas.DataFrame): DataFrame listing out players and their IDs. Must have the following columns:
                Player Name, all PLAYER_IDS. The ALT_PLAYER_IDS columns will be modified; the df is otherwise unmodified.
            master_id_file (str, optional): Filename to master list of players and their different ID formats. Defaults to None.
            pfr_id_filename (str, optional): Filename to pro-football-reference name to ID dictionary. Defaults to None.
            add_missing_pfr (bool, optional): Whether to scrape pro-football-reference for missing PFR IDs. Defaults to False.
            update_master (bool, optional): Whether to add any new IDs found to the master list. Defaults to False.

        Returns:
            pandas.DataFrame: Input DataFrame with any alternate player IDs inserted that were missing and have been found in the master list or elsewhere.
    """  # fmt: skip

    # Reset df index (but hold onto what it was, to be reset later)
    df_index = players_df.index.names
    players_df = players_df.reset_index()

    # Load and quickly update master player ID dataframe
    id_df = __setup_master_id_df(master_id_file, addl_players_df=players_df)

    # Apply player ID map to input roster df
    players_df.loc[:, ALT_PLAYER_IDS] = players_df.apply(
        lambda x: id_df.set_index(PRIMARY_PLAYER_ID).loc[x[PRIMARY_PLAYER_ID], ALT_PLAYER_IDS],
        axis=1,
    )

    # Optionally search for missing data points
    if add_missing_pfr:
        # Log size of the array and number of missing data points
        logger.info(f"Number of Players in Roster: {len(players_df)}")
        logger.info(f"Number of Missing Player IDs: \n{players_df.loc[:, PLAYER_IDS].isna().sum()}")

        # Add missing PFR IDs to id_df
        players_df = __add_missing_pfr_ids(players_df, pfr_id_filename)

        # Log number of missing data points again after updating
        logger.info(f"Number of Missing IDs Remaining: \n{players_df.loc[:, PLAYER_IDS].isna().sum()}")

    # Optionally update the master player ID map and save it to file
    if update_master:
        update_master_player_ids(
            addl_players_df=players_df,
            master_id_file=master_id_file,
            pfr_id_filename=pfr_id_filename,
            add_missing_pfr=False,
            save_data=True,
        )

    # Set index back to how it was
    players_df = players_df.set_index(df_index)

    return players_df


def update_master_player_ids(
    *,
    addl_players_df=None,
    master_id_file=None,
    pfr_id_filename=None,
    add_missing_pfr=False,
    save_data=False,
):
    """Adds potentially new players to the master Player IDs file, and optionally searches for matches to missing IDs in the file.

        Args:
            addl_players_df (pandas.DataFrame, optional): DataFrame listing out players that may not be in the master file. Defaults to None.
            master_id_file (str, optional): Filename to master list of players and their different ID formats. Defaults to None.
            pfr_id_filename (str, optional): Filename to pro-football-reference name to ID dictionary. Defaults to None.
            add_missing_pfr (bool, optional): Whether to scrape pro-football-reference for all missing PFR IDs. Defaults to False.
            save_data (bool, optional): Whether to save the updated DataFrame to a local csv file. Defaults to False.

        Returns:
            pandas.DataFrame: Master Player IDs DataFrame with any new players/IDs added.
    """  # fmt: skip

    id_df = __setup_master_id_df(master_id_file, addl_players_df)

    # Log size of the array and number of missing data points
    logger.info(f"Player IDs Map - Number of Players: {len(id_df)}")
    logger.info(f"Player IDs Map - Number of Missing IDs: \n{id_df.isna().sum()}")

    # Optionally search for missing data points
    if add_missing_pfr:
        # Add missing PFR IDs to id_df
        id_df = __add_missing_pfr_ids(id_df, pfr_id_filename)

        # Log number of missing data points again after updating
        logger.info(f"Number of Missing IDs Remaining: \n{id_df.isna().sum()}")

    # Save updated map to file
    if save_data:
        id_df.to_csv(master_id_file, index=False)

    return id_df


def __setup_master_id_df(master_id_file, addl_players_df=None):
    # DataFrame tracking only player names and IDs
    try:
        id_df = pd.read_csv(master_id_file)
        logger.debug(f"Read Player ID file from {master_id_file}")
    except (FileNotFoundError, ValueError):
        logger.warning("Master Player ID file not found during load process.")
        id_df = pd.DataFrame()

    # Add all player names and IDs from input roster
    if addl_players_df is not None:
        id_df = pd.concat((id_df, addl_players_df[["Player Name", *PLAYER_IDS]]))

    # Merge alt ids for repeated primary IDs and remove duplicate primary IDs
    with pd.option_context("future.no_silent_downcasting", True):
        id_df = (
            id_df.groupby(PRIMARY_PLAYER_ID, dropna=False).apply(lambda x: x.bfill().iloc[0], include_groups=False).reset_index()
        )

    return id_df


def __add_missing_pfr_ids(id_df, pfr_id_filename=None):
    # Search for Missing PFR IDs in two places (in search order):
    # 1. In the locally-saved json file given by pfr_id_filename
    # 2. On pro-football-reference.com
    # Adds any PFR IDs that match the player name to id_df, and updates the local json file.

    # Load previously-found pairs of player names and PFR IDs
    try:
        with open(pfr_id_filename, encoding="utf-8") as file:
            pfr_id_name_dict = json.load(file)
    except (FileNotFoundError, TypeError):
        logger.warning("PFR ID dictionary file not found during load process.")
        pfr_id_name_dict = {}

    pfr_ids_found = []
    n_missing = id_df["pfr_id"].isna().sum()
    for ind, (_, row) in enumerate(id_df.loc[id_df["pfr_id"].isna()].iterrows()):
        player_name = row["Player Name"]
        logger.info(f"({ind + 1} of {n_missing}): {player_name}")
        try:
            # Look for player name in dict of names/IDs already searched
            pfr_ids_found.append(list(pfr_id_name_dict.values())[find_matching_name_ind(player_name, pfr_id_name_dict)])
            logger.debug("id found in json file")
        except TypeError:
            # Search for the ID online
            try:
                pfr_id, new_dict_entries = search_for_missing_pfr_id(player_name)
            except Exception:
                with open(pfr_id_filename, "w", encoding="utf-8") as file:
                    json.dump(pfr_id_name_dict, file)
                raise
            pfr_ids_found.append(pfr_id)
            # Include latest searched name/ID pairs in dict
            pfr_id_name_dict.update(new_dict_entries)
            logger.debug("id found online")
        # Log result
        logger.info(f"{player_name} = {pfr_ids_found[-1]}")

    # Update dataframe
    id_df.loc[id_df["pfr_id"].isna(), "pfr_id"] = pfr_ids_found

    # Update dict file
    try:
        with open(pfr_id_filename, "w", encoding="utf-8") as file:
            json.dump(pfr_id_name_dict, file)
    except (FileNotFoundError, TypeError):
        logger.warning("PFR ID dictionary file not found during save process.")

    return id_df
