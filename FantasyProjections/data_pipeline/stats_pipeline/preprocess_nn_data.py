"""Creates and exports functions to process gathered NFL stats data into a format usable by a Neural Net Fantasy Predictor.

    Functions:
        preprocess_nn_data : Converts NFL stats data from raw statistics to a Neural Network-readable format.
"""  # fmt: skip

import logging

import pandas as pd

from config import data_files_config, stats_config
from config.player_id_config import PRIMARY_PLAYER_ID
from misc.manage_files import create_folders
from misc.stat_utils import normalize_stat

# Set up logger
logger = logging.getLogger("log")


def preprocess_nn_data(midgame_input, final_stats_input, feature_sets, save_folder=None, save_filenames=None):
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

    # Sort by year/week/player/time
    midgame_input = midgame_input.sort_values(
        by=["Year", "Week", "Player ID", "Elapsed Time"],
        ascending=[True, True, True, True],
    )

    # Match inputs (pbp data) to outputs (boxscore data) by index (give each
    # input and corresponding output the same index in their df)
    final_stats_input = (
        final_stats_input.set_index(["Player ID", "Year", "Week"])
        .loc[midgame_input.set_index(["Player ID", "Year", "Week"]).index]
        .reset_index()
    )

    # Break out the columns we want to keep for each dataframe
    id_columns = [
        *stats_config.baseline_data_outputs["id"],
        *[feat.name for feat_set in feature_sets for feat in feat_set.features if "id" in feat.outputs],
    ]
    midgame_columns = [
        *stats_config.baseline_data_outputs["midgame"],
        *[
            feat.name
            for feat_set in feature_sets
            for feat in feat_set.features
            if "midgame" in feat.outputs and not feat.one_hot_encode
        ],
    ]
    final_stats_columns = [
        *stats_config.baseline_data_outputs["final"],
        *[feat.name for feat_set in feature_sets for feat in feat_set.features if "final" in feat.outputs],
    ]

    # Trim each output to only the columns of interest
    id_df = midgame_input[id_columns]
    midgame_input = midgame_input[midgame_columns]
    final_stats_input = final_stats_input[final_stats_columns]

    # Normalize numeric columns to between 0 and 1
    midgame_input = normalize_stat(midgame_input, stats_config.baseline_data_thresholds)
    for feature_set in feature_sets:
        midgame_input = normalize_stat(midgame_input, feature_set.thresholds)
        final_stats_input = normalize_stat(final_stats_input, feature_set.thresholds)

    # One-Hot Encode each non-numeric, relevant pbp field (Player, Team, Position):
    fields_to_encode = [
        *stats_config.baseline_one_hot_columns,
        *[
            feat.name
            for feat_set in feature_sets
            for feat in feat_set.features
            if "midgame" in feat.outputs and feat.one_hot_encode
        ],
    ]
    encoded_fields_df = pd.get_dummies(id_df[fields_to_encode], columns=fields_to_encode, dtype=int)
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
