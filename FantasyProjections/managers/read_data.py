"""Set of functions used to initialize data storage objects for a FantasyProjections scenario.

    Functions:
        read_data_into_dataset : Reads all available input data into a large StatsDataset object.

"""  # fmt:skip

from __future__ import annotations

import logging

import pandas as pd

from config import stats_config
from config.player_id_config import PRIMARY_PLAYER_ID
from misc.dataset import StatsDataset
from misc.stat_utils import normalize_stat

# Set up logger
logger = logging.getLogger("log")


def read_data_into_dataset(features: dict, data_files_config: dict, log_datafiles: bool = True):
    """Reads all available input data into a large StatsDataset object.

        Args:
            log_datafiles (bool, optional): Whether to output status and info to the logger. Defaults to True.

        Returns:
            StatsDataset: dataset named "All" containing all available stats data.

    """  # fmt: skip

    pbp_datafile = data_files_config["stat_files"]["midgame"]
    boxscore_datafile = data_files_config["stat_files"]["final"]

    # Read data files
    pbp_df = pd.read_csv(pbp_datafile, engine="pyarrow")
    boxscore_df = pd.read_csv(boxscore_datafile, engine="pyarrow")

    if log_datafiles:
        logger.info("Data files read")
        for name, file in zip(["pbp", "boxscore"], [pbp_datafile, boxscore_datafile]):
            logger.debug(f"{name}: {file}")

    # Pre-process data before creating dataset
    id_df, pbp_df, boxscore_df, misc_df = preprocess_data(pbp_df, boxscore_df, features)

    # Create dataset containing all data from above files
    all_data = StatsDataset("All", id_df=id_df, pbp_df=pbp_df, boxscore_df=boxscore_df, misc_df=misc_df)

    return all_data


def preprocess_data(
    pbp_df: pd.DataFrame,
    final_stats_df: pd.DataFrame,
    features,
):
    """Converts stats data from raw statistics to a Neural Network-readable format.

        Main steps:
            1. Cleans dataframes (fills in blanks/NaNs as 0, converts all True/False to 1/0, removes non-numeric data)
            2. Matches every row in midgame to the corresponding row in final_stats
            3. Normalizing statistics so that all values are between 0 and 1
            4. Encoding player, team, and opponent IDs as vectors of 0's and 1's (1 corresponds to the correct ID, 0 everywhere else)

        Args:
            data_files_config (dict): Configuration for data files, including paths and filenames.
            midgame_input (pandas.DataFrame | str): Stats accrued over the course of an NFL game for a set of players/games, OR path to csv file containing this data.
            final_stats_input (pandas.DataFrame | str): Stats at the end of an NFL game for a set of players/games, OR path to csv file containing this data.
            feature_sets:
            save_folder (str, optional): folder to save files that can be ingested by a Neural Net Fantasy Predictor. Defaults to None (files will not be saved).
            save_filenames (dict, optional): Filename to use for each neural net input csv. Defaults to filenames in data_files_config.

        Returns:
            pandas.DataFrame: Midgame input data in Neural Net-readable format
            pandas.DataFrame: Final Stats input data in Neural Net-readable format
            pandas.DataFrame: ID (player/game information) input data in Neural Net-readable format

    """  # fmt: skip

    # Format inputs
    pbp_df = pbp_df.rename(columns={PRIMARY_PLAYER_ID: "Player ID"})
    final_stats_df = final_stats_df.rename(columns={PRIMARY_PLAYER_ID: "Player ID"})

    # Replace NaN values with 0
    with pd.option_context("future.no_silent_downcasting", True):
        pbp_df = pbp_df.fillna(0)  # Fill in blank spaces
        final_stats_df = final_stats_df.fillna(0)  # Fill in blank spaces

    # Sort by year/week/player/time
    pbp_df = pbp_df.sort_values(
        by=["Year", "Week", "Player ID", "Elapsed Time"],
        ascending=[True, True, True, True],
    )

    # Match inputs (pbp data) to outputs (boxscore data) by index (give each
    # input and corresponding output the same index in their df)
    final_stats_df = (
        final_stats_df.set_index(["Player ID", "Year", "Week"])
        .loc[pbp_df.set_index(["Player ID", "Year", "Week"]).index]
        .reset_index()
    )

    # Identify the columns we want to keep for each dataframe
    id_columns = list(
        {
            *stats_config.baseline_id_columns,
            *columns_from_features(features, criteria="one_hot_encode", include_string_features=False),
        },
    )
    midgame_columns = columns_from_features(features, "input", criteria="one_hot_encode", invert_criteria=True)
    final_stats_columns = columns_from_features(features, "output", criteria="one_hot_encode", invert_criteria=True)
    # Extract miscellaneous features from the pbp dataframe, including any one-hot encoded columns
    misc_groups = [group for group in features if group not in ["input", "output"]]
    misc_columns = {}
    for group_name in misc_groups:
        misc_columns[group_name] = columns_from_features(
            features,
            group_name,
            criteria="one_hot_encode",
            invert_criteria=True,
        )

    # Trim each output to only the columns of interest
    id_df = pbp_df[id_columns]
    misc_df = pbp_df[[col for group in misc_columns.values() for col in group]]
    pbp_df = pbp_df[midgame_columns]
    final_stats_df = final_stats_df[final_stats_columns]

    # Normalize numeric columns to between 0 and 1
    feature_thresholds = dict(
        zip(
            columns_from_features(features, criteria="thresholds", include_string_features=False),
            columns_from_features(features, criteria="thresholds", return_key="thresholds"),
        ),
    )
    pbp_df = normalize_stat(pbp_df, feature_thresholds)
    final_stats_df = normalize_stat(final_stats_df, feature_thresholds)
    misc_df = normalize_stat(misc_df, feature_thresholds)

    # Name columns of the df
    for group_name, columns in misc_columns.items():
        misc_df = misc_df.rename(columns={col: f"{group_name}_{col}" for col in columns})

    # One-Hot Encode each non-numeric, relevant pbp field (Player, Team, Position):
    input_encoded_features = columns_from_features(features, "input", criteria="one_hot_encode", include_string_features=False)
    output_encoded_features = columns_from_features(features, "output", criteria="one_hot_encode", include_string_features=False)
    misc_encoded_features = {
        feat_group: columns_from_features(features, feat_group, "one_hot_encode", include_string_features=False)
        for feat_group in misc_groups
    }
    if input_encoded_features:
        input_encoded_features_df = pd.get_dummies(id_df[input_encoded_features], columns=input_encoded_features, dtype=int)
        pbp_df = pd.concat((pbp_df, input_encoded_features_df), axis=1)
    if output_encoded_features:
        output_encoded_features_df = pd.get_dummies(id_df[output_encoded_features], columns=output_encoded_features, dtype=int)
        final_stats_df = pd.concat((final_stats_df, output_encoded_features_df), axis=1)
    for group_name, encoded_features in misc_encoded_features.items():
        if encoded_features:
            misc_encoded_features_df = pd.get_dummies(id_df[encoded_features], columns=encoded_features, dtype=int)
            misc_encoded_features_df = misc_encoded_features_df.rename(
                columns={col: f"{group_name}_{col}" for col in misc_encoded_features_df.columns},
            )
            misc_df = pd.concat((misc_df, misc_encoded_features_df), axis=1)

    # Finished pre-processing
    logger.info("Data pre-processed for projections")
    return id_df, pbp_df, final_stats_df, misc_df


def columns_from_features(
    features: dict,
    feature_groups: list | str | None = None,
    criteria: str | None = None,
    invert_criteria: bool = False,
    include_string_features: bool = True,
    return_key: str | None = None,
) -> list:
    """Extracts the columns from the features dictionary, optionally filtering by feature groups.

        Args:
            features (dict): Dictionary containing feature definitions.
            feature_groups (list | str | None, optional): Specific feature groups to extract columns from. Defaults to None (all groups).
            criteria (str | None, optional): Criteria to filter features by. If provided, only features that match the criteria will be included. Defaults to None.
            invert_criteria (bool, optional): If True, inverts the criteria check. Defaults to False.
            include_string_features (bool, optional): If True, includes string features (those without any provided configuration) in the output.
            return_key (str | None, optional): If specified, returns the value associated with this key in the feature dictionary. Defaults to None (returns the feature name).

        Returns:
            list: List of column names extracted from the features.

    """  # fmt: skip

    # Handle list of feature groups to extract columns from
    if isinstance(feature_groups, str):
        feature_groups = [feature_groups]
    if feature_groups is None:
        feature_groups = list(features.keys())

    # If return_key is specified, string features must be excluded
    if return_key is not None:
        include_string_features = False

    # Loop through all feature groups and features within the groups
    columns = []
    for group in feature_groups:
        for feat in features.get(group, []):
            if isinstance(feat, str) and include_string_features:
                # Include simple string features
                columns.append(feat)
            elif isinstance(feat, dict):
                feat_name = next(iter(feat.keys()))
                feat_config = next(iter(feat.values()))
                # Check if criteria is met for the feature
                if (
                    criteria is None
                    or (not invert_criteria and feat_config.get(criteria, False))
                    or (invert_criteria and not feat_config.get(criteria, False))
                ):
                    if return_key is None:
                        columns.append(feat_name)
                    else:
                        columns.append(feat_config.get(return_key))

    return columns


def feature_name(feature):
    if isinstance(feature, dict):
        return next(iter(feature))
    return feature
