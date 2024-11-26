from datetime import datetime
import pandas as pd
from data_pipeline.data_helper_functions import adjust_team_names, cleanup_data, filter_team_weeks
from data_pipeline.roster_filter import generate_roster_filter, apply_roster_filter
from data_pipeline.get_game_info import get_game_info
from data_pipeline.parse_play_by_play import parse_play_by_play
from data_pipeline.process_rosters import process_rosters
from data_pipeline.preprocess_nn_data import preprocess_nn_data
from data_pipeline import team_abbreviations
from misc.manage_files import collect_input_dfs, collect_roster_filter, create_folders


# Flags
SAVE_DATA       = True # Saves data in .csv's (output files specified below)
PROCESS_TO_NN   = True # After saving human-readable data, creates data formatted for Neural Network usage
FILTER_ROSTER   = True # Toggle whether to use filtered list of "relevant" players, vs full rosters for each game
UPDATE_FILTER   = True # Forces re-evaluation of filtered list of players
# Data Inputs
TEAM_NAMES_INPUT    = "all"  # All team names
YEARS               = range(2021, 2025) # All years to process data for
WEEKS               = range(1, 19)      # All weeks to process data for (applies this set to all years in YEARS)
GAME_TIMES          = range(0, 76)      # range(0,76). Alternates: 'all', list of numbers

# Folders
DATA_FOLDER = 'data2/'
INPUT_FOLDER = DATA_FOLDER + 'inputs/'
OUTPUT_FOLDER = DATA_FOLDER + 'stats/'
PRE_PROCESS_FOLDER = DATA_FOLDER + 'to_nn/'
# Files
OUTPUT_FILE_FINAL_STATS = OUTPUT_FOLDER + 'statlines.csv'
OUTPUT_FILE_MIDGAME = OUTPUT_FOLDER + 'pbp_stats.csv'
ROSTER_FILTER_FILE = INPUT_FOLDER + 'Filtered Player List.csv' if FILTER_ROSTER else None
# Input file locations
ONLINE_DATA_SOURCE = 'https://github.com/nflverse/nflverse-data/releases/download/'
online_file_paths = {'pbp': ONLINE_DATA_SOURCE + 'pbp/play_by_play_{0}.csv',
                     'roster': ONLINE_DATA_SOURCE + 'weekly_rosters/roster_weekly_{0}.csv'}
local_file_paths = {'pbp':INPUT_FOLDER + 'play_by_play/play_by_play_{0}.csv',
                    'roster': INPUT_FOLDER + 'rosters/roster_weekly_{0}.csv'}

# Huge output data arrays
midgame_df = pd.DataFrame()
final_stats_df = pd.DataFrame()

# Time the script
start_time = datetime.now()

# Load dataframes from files/online
input_dfs = collect_input_dfs(YEARS, WEEKS, local_file_paths, online_file_paths, online_avail=True)
# Load optional roster filter
filter_df, filter_load_success = collect_roster_filter(FILTER_ROSTER, UPDATE_FILTER, ROSTER_FILTER_FILE)

for year, (pbp_df, roster_df) in zip(YEARS,input_dfs):
    print(f"------------------{year}------------------")
    adjust_team_names((team_abbreviations.pbp_abbrevs,
                       team_abbreviations.boxscore_website_abbrevs,
                       team_abbreviations.roster_website_abbrevs),
                       year)
    TEAM_NAMES = list(team_abbreviations.roster_website_abbrevs.keys()) if TEAM_NAMES_INPUT == "all" else TEAM_NAMES_INPUT

    # Process team records, game sites, and other game info for every game of the year, for every team
    all_game_info_df = get_game_info(year, pbp_df)

    # Gather team roster for all teams, all weeks of input year
    all_rosters_df = process_rosters(roster_df, WEEKS, filter_df)

    for full_team_name in TEAM_NAMES:
        team_abbrev = team_abbreviations.pbp_abbrevs[full_team_name]
        print(f"---------{full_team_name}---------")

        # All regular season weeks where games have been played by the team
        # (excludes byes, future weeks)
        weeks_with_games, weeks_with_players = filter_team_weeks(all_game_info_df, all_rosters_df, full_team_name, year)

        for week in WEEKS:
            if week not in weeks_with_games:
                print(f"Week {week} - no game found")
            elif week not in weeks_with_players:
                print(f"Week {week} - no players found")
            else:
                print(f"Week {week}")
                # Game info, roster for specific team & week
                game_info = all_game_info_df.loc[(full_team_name, year, week)]
                roster_df = all_rosters_df.loc[
                    (team_abbreviations.pbp_abbrevs[full_team_name], week)
                ].set_index("Name")

                # Parse play-by-play to get running stats for each player
                [midgame_df_one_game, final_stats_df_one_game] = parse_play_by_play(
                    pbp_df, roster_df, game_info, GAME_TIMES)

                # Append results from this game to df of results from all games
                midgame_df = pd.concat([midgame_df, midgame_df_one_game])
                final_stats_df = pd.concat([final_stats_df, final_stats_df_one_game])

        # Display running total of data rows in df
        print(f"\tdata rows: {midgame_df.shape[0]}")

# If roster filter could not be found/applied before processing,
# generate a roster filter file now and apply it to the data
if FILTER_ROSTER and (not (filter_load_success) or UPDATE_FILTER):
    filter_df = generate_roster_filter(input_dfs, final_stats_df, ROSTER_FILTER_FILE)
    midgame_df, final_stats_df = apply_roster_filter(midgame_df, final_stats_df, filter_df)

# Organize dataframes
midgame_df, final_stats_df = cleanup_data(midgame_df, final_stats_df)
print(midgame_df)

# Save raw data
if SAVE_DATA:
    create_folders(OUTPUT_FOLDER)
    final_stats_df.to_csv(OUTPUT_FILE_FINAL_STATS)
    midgame_df.to_csv(OUTPUT_FILE_MIDGAME)

# Generate/save data in a format readable into the Neural Net
if PROCESS_TO_NN:
    preprocess_nn_data(midgame_input=midgame_df, final_stats_input=final_stats_df, save_data=SAVE_DATA, save_folder=PRE_PROCESS_FOLDER)

print(f"Done! Elapsed Time: {(datetime.now() - start_time).total_seconds()} seconds")
