"""Creates and exports functions to process gathered NFL stats data into a format usable by a Neural Net Fantasy Predictor.

    Functions:
        preprocess_nn_data : Converts NFL stats data from raw statistics to a Neural Network-readable format.
"""  # fmt: skip

import logging

import pandas as pd

from config import data_files_config, stats_config
from config.player_id_config import ALT_PLAYER_IDS, PRIMARY_PLAYER_ID
from misc.manage_files import create_folders
from misc.stat_utils import normalize_stat

# Set up logger
logger = logging.getLogger("log")


def preprocess_nn_data(midgame_input, final_stats_input, save_folder=None, save_filenames=None):
    """Converts NFL stats data from raw statistics to a Neural Network-readable format.

        Main steps:
            1. Cleans dataframes (fills in blanks/NaNs as 0, converts all True/False to 1/0, removes non-numeric data)
            2. Matches every row in midgame to the corresponding row in final_stats
            3. Normalizing statistics so that all values are between 0 and 1
            4. Encoding player, team, and opponent IDs as vectors of 0's and 1's (1 corresponds to the correct ID, 0 everywhere else)

        Args:
            midgame_input (pandas.DataFrame | str): Stats accrued over the course of an NFL game for a set of players/games, OR path to csv file containing this data.
            final_stats_input (pandas.DataFrame | str): Stats at the end of an NFL game for a set of players/games, OR path to csv file containing this data.
            save_folder (str, optional): folder to save files that can be ingested by a Neural Net Fantasy Predictor. Defaults to None (files will not be saved).
            save_filenames (dict, optional): Filename to use for each neural net input csv. Defaults to filenames in data_files_config.

        Returns:
            pandas.DataFrame: Midgame input data in Neural Net-readable format
            pandas.DataFrame: Final Stats input data in Neural Net-readable format
            pandas.DataFrame: ID (player/game information) input data in Neural Net-readable format

    """  # fmt: skip

    # Optional save_filenames input
    if not save_filenames:
        save_filenames = data_files_config.NN_STAT_FILES

    # Read files if raw dataframes are not passed in
    if not isinstance(midgame_input, pd.DataFrame):
        midgame_input = pd.read_csv(midgame_input, low_memory=False)
    else:
        midgame_input = midgame_input.reset_index(drop=False).rename(columns={PRIMARY_PLAYER_ID: "Player ID"})
    if not isinstance(final_stats_input, pd.DataFrame):
        final_stats_input = pd.read_csv(final_stats_input)
    else:
        final_stats_input = final_stats_input.reset_index(drop=False).rename(columns={PRIMARY_PLAYER_ID: "Player ID"})

    # A few things to prep the data for use in the NN
    with pd.option_context("future.no_silent_downcasting", True):
        midgame_input = midgame_input.fillna(0)  # Fill in blank spaces
        final_stats_input = final_stats_input.fillna(0)  # Fill in blank spaces
    midgame_input["Site"] = pd.to_numeric(midgame_input["Site"] == "Home")  # Convert Site to 1/0
    midgame_input["Possession"] = pd.to_numeric(midgame_input["Possession"])  # Convert Possession to 1/0

    # Only keep numeric data (non-numeric columns will be stripped out later in pre-processing)
    ids = ["Player ID", *ALT_PLAYER_IDS]
    id_columns = [*ids, "Player Name", "Year", "Week", "Team", "Opponent", "Position", "Elapsed Time"]
    midgame_numeric_columns = [
        "Elapsed Time",
        "Team Score",
        "Opp Score",
        "Possession",
        "Field Position",
        *stats_config.default_stat_list,
        "injury_status",
        "Age",
        "Site",
        "Team Wins",
        "Team Losses",
        "Team Ties",
        "Opp Wins",
        "Opp Losses",
        "Opp Ties",
    ]
    final_stats_numeric_columns = stats_config.default_stat_list

    # Sort by year/week/team/player
    midgame_input = midgame_input.sort_values(
        by=["Year", "Week", "Team", "Player ID"],
        ascending=[True, True, True, True],
    )

    # Match inputs (pbp data) to outputs (boxscore data) by index (give each
    # input and corresponding output the same index in their df)
    final_stats_input = (
        final_stats_input.set_index(["Player ID", "Year", "Week"])
        .loc[midgame_input.set_index(["Player ID", "Year", "Week"]).index]
        .reset_index()
    )

    # Keep identifying info in a separate dataframe
    id_df = midgame_input[id_columns]

    # Strip out non-numeric columns, and normalize numeric columns to between 0 and 1
    midgame_input = midgame_input[midgame_numeric_columns]
    midgame_input = normalize_stat(midgame_input)
    final_stats_input = final_stats_input[final_stats_numeric_columns]
    final_stats_input = normalize_stat(final_stats_input)

    # One-Hot Encode each non-numeric, relevant pbp field (Player, Team, Position):
    fields = ["Position", "Player ID", "Team", "Opponent"]
    encoded_fields_df = pd.get_dummies(id_df[fields], columns=fields, dtype=int)
    midgame_input = pd.concat((midgame_input, encoded_fields_df), axis=1)
    logger.info("Data pre-processed for projections")

    # Save data
    if save_folder is not None:
        create_folders(save_folder)
        logger.info("Saving pre-processed NN data")
        midgame_input.to_csv(f"{save_folder}{save_filenames['midgame']}", index=False)
        final_stats_input.to_csv(f"{save_folder}{save_filenames['final']}", index=False)
        id_df.to_csv(f"{save_folder}{save_filenames['id']}", index=False)
        logger.info(f"Saved pre-processed data to {save_folder}")

    return midgame_input, final_stats_input, id_df
