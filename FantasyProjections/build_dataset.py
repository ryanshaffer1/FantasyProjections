"""Gathers data to be used in Fantasy Football stat prediction.
Pulls data from local files where possible, and from online repository when necessary (saving locally for next time). Parses inputs to determine the
play-by-play performance of all NFL players or a filtered subset (filtered by highest-scoring in Fantasy Football) over all games within a specified
time-frame. Optionally normalizes/pre-processes the data for use in a Neural Net predictor.

This module is a script to be run alone. Before running, the packages in requirements.txt must be installed. Use the following terminal command:
> pip install -u requirements.txt
"""  # fmt: skip

# fmt: skip
import logging
import logging.config
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from config import data_files_config
from config.log_config import LOGGING_CONFIG
from data_pipeline.stats_pipeline.preprocess_nn_data import preprocess_nn_data
from data_pipeline.stats_pipeline.roster_filter import apply_roster_filter, generate_roster_filter
from data_pipeline.stats_pipeline.seasonal_stats_collector import SeasonalStatsCollector
from data_pipeline.stats_pipeline.validate_parsed_data import validate_parsed_data
from data_pipeline.utils.data_helper_functions import clean_stats_data
from misc.manage_files import collect_roster_filter, create_folders, move_logfile

# Flags
SAVE_DATA = True  # Saves data in .csv's (output files specified below)
PROCESS_TO_NN = True  # After saving human-readable data, creates data formatted for Neural Network usage
FILTER_ROSTER = True  # Toggle whether to use filtered list of "relevant" players, vs full rosters for each game
UPDATE_FILTER = False  # Forces re-evaluation of filtered list of players
VALIDATE_PARSING = True  # Gathers true box scores from the internet to confirm logic in play-by-play parsing is correct
SCRAPE_MISSING = True  # Scrapes Pro-Football-Reference.com to gather true player stats for any missing players
# Data Inputs
TEAM_NAMES = "all"  # All team names
YEARS = range(2020, 2025)  # All years to process data for
WEEKS = range(1, 19)  # All weeks to process data for (applies this set to all years in YEARS)
GAME_TIMES = range(76)  # range(0,76). Alternates: 'all', list of numbers


# Set up logger
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")

# Start
start_time = datetime.now().astimezone()
logger.info("Starting Program")

# Log user inputs
logger.debug(
    f"FLAGS: \nSAVE_DATA={SAVE_DATA} \nPROCESS_TO_NN={PROCESS_TO_NN} \nFILTER_ROSTER={FILTER_ROSTER} \nUPDATE_FILTER={UPDATE_FILTER}\
            \nVALIDATE_PARSING={VALIDATE_PARSING} \nSCRAPE_MISSING={SCRAPE_MISSING}",
)
logger.debug(f"DATA INPUTS: \nTEAM_NAMES={TEAM_NAMES} \nYEARS={YEARS} \nWEEKS={WEEKS} \nGAME_TIMES={GAME_TIMES}")

# Files to optionally load
ROSTER_FILTER_FILE = data_files_config.ROSTER_FILTER_FILE if FILTER_ROSTER else None
# Files to save
if SAVE_DATA:
    ROSTER_SAVE_FILE = ROSTER_FILTER_FILE
    PRE_PROCESS_FOLDER = data_files_config.PRE_PROCESS_FOLDER
else:
    ROSTER_SAVE_FILE = None
    PRE_PROCESS_FOLDER = None

# Huge output data arrays
midgame_df = pd.DataFrame()
final_stats_df = pd.DataFrame()
urls_df = pd.DataFrame()

# Load optional roster filter
filter_df, filter_load_success = collect_roster_filter(FILTER_ROSTER, UPDATE_FILTER, ROSTER_FILTER_FILE)
logger.info(f"Filter Load Success: {filter_load_success}; Filter file: {ROSTER_FILTER_FILE}")


# Process NFL data one year at a time
for year in YEARS:
    logger.info(f"--------------- {year} ---------------")
    # Create SeasonalData object, which automatically processes all data for that year
    seasonal_data = SeasonalStatsCollector(
        year=year,
        team_names=TEAM_NAMES,
        weeks=WEEKS,
        game_times=GAME_TIMES,
        filter_df=filter_df,
    )
    # Concatenate results from current year to remaining years
    midgame_df = pd.concat((midgame_df, seasonal_data.midgame_df))
    final_stats_df = pd.concat((final_stats_df, seasonal_data.final_stats_df))
    urls_df = pd.concat((urls_df, seasonal_data.all_game_info_df["PFR URL"]))

    logger.info(f"{year} processing complete.")
logger.info("Data collection complete.")
logger.info(f"Total midgame data rows: {midgame_df.shape[0]}")

# If roster filter could not be found/applied before processing,
# generate a roster filter file now and apply it to the data
if FILTER_ROSTER and (not (filter_load_success) or UPDATE_FILTER):
    logger.info("Generating new filter.")
    filter_df = generate_roster_filter(seasonal_data.raw_rosters_df, final_stats_df, ROSTER_SAVE_FILE)
    midgame_df, final_stats_df = apply_roster_filter(midgame_df, final_stats_df, filter_df)

# Organize dataframes
midgame_df, final_stats_df = clean_stats_data(midgame_df, final_stats_df)

# Save raw data
if SAVE_DATA:
    create_folders(data_files_config.OUTPUT_FOLDER)
    midgame_df.to_csv(data_files_config.OUTPUT_FILE_MIDGAME)
    logger.info(f"Saved midgame stats to {data_files_config.OUTPUT_FILE_MIDGAME}.")
    final_stats_df.to_csv(data_files_config.OUTPUT_FILE_FINAL_STATS)
    logger.info(f"Saved final stats to {data_files_config.OUTPUT_FILE_FINAL_STATS}.")

# Optionally validate that parsed statlines match statlines found on the internet
if VALIDATE_PARSING:
    logger.info("Validating Parsed Data.")
    validate_parsed_data(final_stats_df, urls_df, scrape=SCRAPE_MISSING, save_data=SAVE_DATA)

# Generate/save data in a format readable into the Neural Net
if PROCESS_TO_NN:
    logger.info("Pre-processing data for use in Neural Net.")
    preprocess_nn_data(midgame_input=midgame_df, final_stats_input=final_stats_df, save_folder=PRE_PROCESS_FOLDER)

# Finish up
logger.info(f"Program complete. Elapsed Time: {(datetime.now().astimezone() - start_time).total_seconds()} seconds")
logging.shutdown()
move_logfile("logfile.log", data_files_config.DATA_FOLDER)
plt.show()
