"""Gathers data to be used in Fantasy Football stat prediction.
    Pulls data from local files where possible, and from online repository when necessary (saving locally for next time). Parses inputs to determine the
    play-by-play performance of all NFL players or a filtered subset (filtered by highest-scoring in Fantasy Football) over all games within a specified
    time-frame. Optionally normalizes/pre-processes the data for use in a Neural Net predictor.

    This module is a script to be run alone. Before running, the packages in requirements.txt must be installed. Use the following terminal command:
    > pip install -u requirements.txt
"""  # fmt: skip

import logging
import logging.config
from datetime import datetime

import pandas as pd
from config import data_files_config
from config.log_config import LOGGING_CONFIG
from data_pipeline.odds_pipeline.seasonal_odds_collector import SeasonalOddsCollector
from misc.manage_files import collect_roster_filter, create_folders, move_logfile

# Flags
SAVE_DATA = True  # Saves data in .csv's (output files specified below)
PROCESS_TO_NN = True  # After saving human-readable data, creates data formatted for Neural Network usage
FILTER_ROSTER = False  # Toggle whether to use filtered list of "relevant" players, vs full rosters for each game
UPDATE_FILTER = False  # Forces re-evaluation of filtered list of players
VALIDATE_PARSING = True  # Gathers true box scores from the internet to confirm logic in play-by-play parsing is correct
SCRAPE_MISSING = True  # Scrapes Pro-Football-Reference.com to gather true player stats for any missing players
# Data Inputs
TEAM_NAMES = "all"  # All team names
YEARS = range(2023, 2024)  # All years to process data for
WEEKS = range(1, 18)  # All weeks to process data for (applies this set to all years in YEARS)


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
logger.debug(f"DATA INPUTS: \nTEAM_NAMES={TEAM_NAMES} \nYEARS={YEARS} \nWEEKS={WEEKS}")

# Files to optionally load
ROSTER_FILTER_FILE = data_files_config.ROSTER_FILTER_FILE if FILTER_ROSTER else None
# Files to save
if SAVE_DATA:
    ROSTER_SAVE_FILE = ROSTER_FILTER_FILE
    PRE_PROCESS_FOLDER = data_files_config.PRE_PROCESS_FOLDER
else:
    ROSTER_SAVE_FILE = None
    PRE_PROCESS_FOLDER = None

# Initialize output data array
try:
    odds_df = pd.read_csv(data_files_config.ODDS_FILE).set_index("Player ID")
    logger.info(f"Pre-existing odds data rows: {odds_df.shape[0]}")
except FileNotFoundError:
    odds_df = pd.DataFrame()
    logger.info("No pre-existing odds file found.")


# Load optional roster filter
filter_df, filter_load_success = collect_roster_filter(FILTER_ROSTER, UPDATE_FILTER, ROSTER_FILTER_FILE)
logger.info(f"Filter Load Success: {filter_load_success}; Filter file: {ROSTER_FILTER_FILE}")


# Process NFL data one year at a time
for year in YEARS:
    logger.info(f"--------------- {year} ---------------")
    # Create SeasonalData object, which automatically processes all data for that year
    seasonal_data = SeasonalOddsCollector(
        year=year,
        team_names=TEAM_NAMES,
        weeks=WEEKS,
        filter_df=filter_df,
        odds_file=data_files_config.ODDS_FILE,
        player_props=["Rec Yds", "Rush Yds", "Pass Yds"],
    )
    # Concatenate results from current year to remaining years
    odds_df = pd.concat((odds_df, seasonal_data.odds_df))

    logger.info(f"{year} processing complete.")
logger.info("Data collection complete.")
# Clean up
odds_df = odds_df.drop_duplicates(keep="first")
logger.info(f"Total odds data rows: {odds_df.shape[0]}")


# Save raw data
if SAVE_DATA:
    create_folders(data_files_config.ODDS_FOLDER)
    odds_df.to_csv(data_files_config.ODDS_FILE)

# Finish up
logger.info(f"Program complete. Elapsed Time: {(datetime.now().astimezone() - start_time).total_seconds()} seconds")
logging.shutdown()
move_logfile("logfile.log", data_files_config.DATA_FOLDER)
