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


def read_data_into_dataset(log_datafiles=True):
    # Read data files
    pbp_df = pd.read_csv(PBP_DATAFILE, engine="pyarrow")
    boxscore_df = pd.read_csv(BOXSCORE_DATAFILE, engine="pyarrow")
    id_df = pd.read_csv(ID_DATAFILE, engine="pyarrow")

    if log_datafiles:
        logger.info("Data files read")
        for name, file in zip(["pbp", "boxscore", "IDs"], [PBP_DATAFILE, BOXSCORE_DATAFILE, ID_DATAFILE]):
            logger.debug(f"{name}: {file}")

    # Create dataset containing all data from above files
    all_data = StatsDataset("All", id_df=id_df, pbp_df=pbp_df, boxscore_df=boxscore_df)

    return all_data
