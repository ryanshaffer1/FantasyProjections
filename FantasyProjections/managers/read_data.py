"""Set of functions used to initialize data storage objects for a FantasyProjections scenario.

    Functions:
        read_data_into_dataset : Reads all available input data into a large StatsDataset object.

"""  # fmt:skip

import logging
import os

import pandas as pd

from misc.dataset import StatsDataset

# Set up logger
logger = logging.getLogger("log")


def read_data_into_dataset(features: dict, data_files_config: dict, log_datafiles: bool = True):
    """Reads all available input data into a large StatsDataset object.

        Args:
            log_datafiles (bool, optional): Whether to output status and info to the logger. Defaults to True.

        Returns:
            StatsDataset: dataset named "All" containing all available stats data.

    """  # fmt: skip

    pbp_datafile = os.path.join(data_files_config["pre_process_folder"], data_files_config["nn_stat_files"]["midgame"])
    boxscore_datafile = os.path.join(data_files_config["pre_process_folder"], data_files_config["nn_stat_files"]["final"])
    id_datafile = os.path.join(data_files_config["pre_process_folder"], data_files_config["nn_stat_files"]["id"])

    # Read data files
    pbp_df = pd.read_csv(pbp_datafile, engine="pyarrow")
    boxscore_df = pd.read_csv(boxscore_datafile, engine="pyarrow")
    id_df = pd.read_csv(id_datafile, engine="pyarrow")

    if log_datafiles:
        logger.info("Data files read")
        for name, file in zip(["pbp", "boxscore", "IDs"], [pbp_datafile, boxscore_datafile, id_datafile]):
            logger.debug(f"{name}: {file}")

    # Filter dataframes to just the desired features (and order dataframe according to features)
    input_features, output_features, misc_features = read_features(features, pbp_df, boxscore_df)
    pbp_df_filtered = pbp_df[input_features]
    boxscore_df_filtered = boxscore_df[output_features]
    misc_df = pd.DataFrame(index=pbp_df.index)
    for feat_group, feat_list in misc_features.items():
        misc_df = pd.concat((misc_df, pbp_df[feat_list]), axis=1)
        misc_df = misc_df.rename(columns={col: f"{feat_group}_{col}" for col in feat_list})

    # Create dataset containing all data from above files
    all_data = StatsDataset("All", id_df=id_df, pbp_df=pbp_df_filtered, boxscore_df=boxscore_df_filtered, misc_df=misc_df)

    return all_data


def read_features(features: dict, pbp_df: pd.DataFrame, boxscore_df: pd.DataFrame) -> tuple[list, list, dict]:
    # Extract input features from the pbp dataframe, including any one-hot encoded columns
    input_features = []
    for feat_name in features.get("input", []):
        # Add the feature name if it is not one-hot encoded
        if feat_name in pbp_df.columns:
            input_features.append(feat_name)
        # Add any one-hot encoded columns related to the feature (will evaluate to an empty list if not encoded)
        all_encoded_cols = pbp_df.filter(like=f"{feat_name}_", axis=1).columns.to_list()
        input_features += all_encoded_cols

    # Extract output features from the boxscore dataframe, including any one-hot encoded columns
    output_features = []
    for feat_name in features.get("output", []):
        # Add the feature name if it is not one-hot encoded
        if feat_name in boxscore_df.columns:
            output_features.append(feat_name)
        # Add any one-hot encoded columns related to the feature (will evaluate to an empty list if not encoded)
        all_encoded_cols = boxscore_df.filter(like=f"{feat_name}_", axis=1).columns.to_list()
        output_features += all_encoded_cols

    # Extract miscellaneous features from the pbp dataframe, including any one-hot encoded columns
    misc_features = {}
    for feat_group, feat_list in features.items():
        if feat_group in ["input", "output"]:
            continue
        misc_features[feat_group] = []
        for feat_name in feat_list:
            # Add the feature name if it is not one-hot encoded
            if feat_name in pbp_df.columns:
                misc_features[feat_group].append(feat_name)
            # Add any one-hot encoded columns related to the feature (will evaluate to an empty list if not encoded)
            all_encoded_cols = pbp_df.filter(like=f"{feat_name}_", axis=1).columns.to_list()
            misc_features[feat_group] += all_encoded_cols

    return input_features, output_features, misc_features
