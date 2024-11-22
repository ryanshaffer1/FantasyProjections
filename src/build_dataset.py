from datetime import datetime
import pandas as pd
from data_pipeline.process_rosters import process_rosters
from data_pipeline.get_game_info import get_game_info
from data_pipeline.parse_play_by_play import parse_play_by_play
from data_pipeline.data_helper_functions import adjust_team_names
from data_pipeline.preprocess_nn_data import preprocess_nn_data
from data_pipeline import team_abbreviations
from misc.manage_files import collect_input_dfs


# Inputs
SAVE_DATA = False
PROCESS_TO_NN = True
# Toggle whether to use filtered list of "relevant" players, vs full
# rosters for each game
FILTER_ROSTER = True
TEAM_NAMES = "all"  # All team names
YEARS = range(2021, 2025)
WEEKS = range(1, 19)
GAME_TIMES = range(0, 76)  # range(0,76). Alternates: 'all', list of numbers
# Files
OUTPUT_FILE_BOX = "data/statlines.csv"
OUTPUT_FILE_MIDGAME = "data/pbp_stats.csv"
ROSTER_FILTER_FILE = "data/inputs/Filtered Player List.csv" if FILTER_ROSTER else None

# Input file locations
ONLINE_REPO = 'https://github.com/nflverse/nflverse-data/releases/download/'
online_file_paths = {'pbp': ONLINE_REPO+'pbp/play_by_play_{0}.csv',
                     'roster': ONLINE_REPO+'weekly_rosters/roster_weekly_{0}.csv'}
LOCAL_REPO = 'data/inputs/'
local_file_paths = {'pbp':LOCAL_REPO+'play_by_play/play_by_play_{0}.csv',
                    'roster': LOCAL_REPO+'rosters/roster_weekly_{0}.csv'}

# Huge output data arrays
all_stats_df = pd.DataFrame()
all_boxes_df = pd.DataFrame()

# Dummy variable for a variable that might be overwritten later
TEAM_NAMES_INPUT = TEAM_NAMES

# Time the script
start_time = datetime.now()

for year in YEARS:
    print(f"------------------{year}------------------")
    adjust_team_names((team_abbreviations.pbp_abbrevs,
                       team_abbreviations.boxscore_website_abbrevs,
                       team_abbreviations.roster_website_abbrevs),
                       year)
    if TEAM_NAMES_INPUT == "all":
        TEAM_NAMES = list(team_abbreviations.roster_website_abbrevs.keys())
        TEAM_NAMES.sort()

    # Load dataframes from files/online
    (pbp_df, roster_df) = collect_input_dfs(year, WEEKS, local_file_paths, online_file_paths, online_avail=True)

    # Process team records, game sites, and other game info for every game of
    # the year, for every team
    all_game_info_df = get_game_info(year, pbp_df)

    # Gather team roster for all teams, all weeks of input year
    all_rosters_df = process_rosters(roster_df, WEEKS, ROSTER_FILTER_FILE)

    for full_team_name in TEAM_NAMES:
        print(f"---------{full_team_name}---------")

        # All regular season weeks where games have been played by the team
        # (excludes byes, future weeks)
        weeks_with_games = list(all_game_info_df.loc[(full_team_name, year)].index)
        weeks_with_players = (
            all_rosters_df.loc[(team_abbreviations.pbp_abbrevs[full_team_name])]
            .index.unique()
            .to_list()
        )
        for week in WEEKS:
            if week not in weeks_with_games:
                print(f"Week {week} - no game found")
            elif week not in weeks_with_players:
                print(f"Week {week} - no players found")
            else:
                print(f"Week {week}")
                # Game info for specific team & week
                game_info_ser = all_game_info_df.loc[(full_team_name, year, week)]

                # Roster for specific team & week
                roster_df = all_rosters_df.loc[
                    (team_abbreviations.pbp_abbrevs[full_team_name], week)
                ].set_index("Name")

                # Parse play-by-play to get running stats for each player
                [game_stats_df, box_score_df] = parse_play_by_play(
                    full_team_name, pbp_df, roster_df, game_info_ser, GAME_TIMES)

                all_stats_df = pd.concat([all_stats_df, game_stats_df])
                all_boxes_df = pd.concat([all_boxes_df, box_score_df])
        print(f"\tdata rows: {all_stats_df.shape[0]}")

# Organize dfs
all_boxes_df = (
    all_boxes_df.reset_index().set_index(["Player", "Year", "Week"]).sort_index()
)
list_of_stats = [
    "Pass Att",
    "Pass Cmp",
    "Pass Yds",
    "Pass TD",
    "Int",
    "Rush Att",
    "Rush Yds",
    "Rush TD",
    "Rec",
    "Rec Yds",
    "Rec TD",
    "Fmb",
    "Fantasy Score",
]
all_boxes_df = all_boxes_df[["Team", "Opponent", "Position", "Age"] + list_of_stats]
all_stats_df = all_stats_df.reset_index().set_index(
    ["Player", "Year", "Week", "Elapsed Time"]
)
print(all_stats_df)

if SAVE_DATA:
    all_boxes_df.to_csv(OUTPUT_FILE_BOX)
    all_stats_df.to_csv(OUTPUT_FILE_MIDGAME)

print(f"Elapsed Time: {(datetime.now() - start_time).total_seconds()} seconds")

if PROCESS_TO_NN:
    preprocess_nn_data(pbp_df=all_stats_df, boxscore_df=all_boxes_df, save_data=SAVE_DATA)
