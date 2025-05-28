"""Set of functions used to initialize data storage objects for a FantasyProjections scenario.

    Functions:
        read_data_into_dataset : Reads all available input data into a large StatsDataset object.

"""  # fmt:skip

import logging

import pandas as pd

from config import data_files_config
from misc.dataset import StatsDataset

# Set up logger
logger = logging.getLogger("log")

# Neural Net Data files
PBP_DATAFILE = data_files_config.PRE_PROCESS_FOLDER + data_files_config.NN_STAT_FILES["midgame"]
BOXSCORE_DATAFILE = data_files_config.PRE_PROCESS_FOLDER + data_files_config.NN_STAT_FILES["final"]
ID_DATAFILE = data_files_config.PRE_PROCESS_FOLDER + data_files_config.NN_STAT_FILES["id"]


def read_data_into_dataset(features: dict, log_datafiles: bool = True):
    """Reads all available input data into a large StatsDataset object.

        Args:
            log_datafiles (bool, optional): Whether to output status and info to the logger. Defaults to True.

        Returns:
            StatsDataset: dataset named "All" containing all available stats data.

    """  # fmt: skip

    # Read data files
    pbp_df = pd.read_csv(PBP_DATAFILE, engine="pyarrow")
    boxscore_df = pd.read_csv(BOXSCORE_DATAFILE, engine="pyarrow")
    id_df = pd.read_csv(ID_DATAFILE, engine="pyarrow")

    if log_datafiles:
        logger.info("Data files read")
        for name, file in zip(["pbp", "boxscore", "IDs"], [PBP_DATAFILE, BOXSCORE_DATAFILE, ID_DATAFILE]):
            logger.debug(f"{name}: {file}")

    # Filter dataframes to just the desired features (and order dataframe according to features)
    input_features = read_features(features, pbp_df)
    pbp_df = pbp_df[input_features]

    # Create dataset containing all data from above files
    all_data = StatsDataset("All", id_df=id_df, pbp_df=pbp_df, boxscore_df=boxscore_df)

    return all_data


def read_features(features: dict, pbp_df: pd.DataFrame) -> list:
    input_features = []
    for feat_name, feat_specs in features.get("input", {}).items():
        if feat_specs is None or not feat_specs.get("one_hot_encoded", False):
            input_features.append(feat_name)
        else:
            all_encoded_cols = pbp_df.filter(like=f"{feat_name}_", axis=1).columns.to_list()
            input_features += all_encoded_cols

    return input_features
