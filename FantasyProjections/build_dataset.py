"""Gathers data to be used in Fantasy Football stat prediction. 
    Pulls data from local files where possible, and from online repository when necessary (saving locally for next time). Parses inputs to determine the 
    play-by-play performance of all NFL players or a filtered subset (filtered by highest-scoring in Fantasy Football) over all games within a specified
    time-frame. Optionally normalizes/pre-processes the data for use in a Neural Net predictor.
    
    This module is a script to be run alone. Before running, the packages in requirements.txt must be installed. Use the following terminal command:
    > pip install -u requirements.txt
"""

from datetime import datetime
import pandas as pd
from config import data_files_config
from data_pipeline.seasonal_data import SeasonalDataCollector
from data_pipeline.data_helper_functions import cleanup_data
from data_pipeline.roster_filter import generate_roster_filter, apply_roster_filter
from data_pipeline.preprocess_nn_data import preprocess_nn_data
from misc.manage_files import collect_roster_filter, create_folders


# Flags
SAVE_DATA       = False # Saves data in .csv's (output files specified below)
PROCESS_TO_NN   = True # After saving human-readable data, creates data formatted for Neural Network usage
FILTER_ROSTER   = True # Toggle whether to use filtered list of "relevant" players, vs full rosters for each game
UPDATE_FILTER   = False # Forces re-evaluation of filtered list of players
# Data Inputs
TEAM_NAMES_INPUT    = 'all'  # All team names
YEARS               = range(2022, 2024) # All years to process data for
WEEKS               = range(1, 4)      # All weeks to process data for (applies this set to all years in YEARS)
GAME_TIMES          = range(0, 76)      # range(0,76). Alternates: 'all', list of numbers


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

# Time the script
start_time = datetime.now()

# Load optional roster filter
filter_df, filter_load_success = collect_roster_filter(FILTER_ROSTER, UPDATE_FILTER, ROSTER_FILTER_FILE)

# Process NFL data one year at a time
for year in YEARS:
    print(f"------------------{year}------------------")
    # Create SeasonalData object
    seasonal_data = SeasonalDataCollector(year=year, team_names=TEAM_NAMES_INPUT, weeks=WEEKS,
                                 game_times=GAME_TIMES, filter_df=filter_df)
    # Concatenate results from current year to remaining years
    midgame_df = pd.concat((midgame_df, seasonal_data.midgame_df))
    final_stats_df = pd.concat((final_stats_df, seasonal_data.final_stats_df))

# If roster filter could not be found/applied before processing,
# generate a roster filter file now and apply it to the data
if FILTER_ROSTER and (not (filter_load_success) or UPDATE_FILTER):
    filter_df = generate_roster_filter(seasonal_data.raw_rosters_df, final_stats_df, ROSTER_SAVE_FILE)
    midgame_df, final_stats_df = apply_roster_filter(midgame_df, final_stats_df, filter_df)

# Organize dataframes
midgame_df, final_stats_df = cleanup_data(midgame_df, final_stats_df)
print(midgame_df)

# Save raw data
if SAVE_DATA:
    create_folders(data_files_config.OUTPUT_FOLDER)
    final_stats_df.to_csv(data_files_config.OUTPUT_FILE_FINAL_STATS)
    midgame_df.to_csv(data_files_config.OUTPUT_FILE_MIDGAME)

# Generate/save data in a format readable into the Neural Net
if PROCESS_TO_NN:
    preprocess_nn_data(midgame_input=midgame_df, final_stats_input=final_stats_df, save_folder=PRE_PROCESS_FOLDER)

print(f"Done! Elapsed Time: {(datetime.now() - start_time).total_seconds()} seconds")
