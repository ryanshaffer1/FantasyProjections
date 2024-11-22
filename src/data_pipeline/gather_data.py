from datetime import datetime
import pandas as pd
from process_rosters import process_rosters
from get_game_info import get_game_info
from parse_play_by_play import parse_play_by_play
from data_helper_functions import adjust_team_names
import team_abbreviations

# Inputs
SAVE_DATA = True
# Toggle whether to keep the in-game stats, or only track box scores
# (final stats)
TRACK_MIDGAME_STATS = True
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

# Huge output data arrays
all_stats_df = pd.DataFrame()
all_boxes_df = pd.DataFrame()

# Dummy variable for a variable that might be overwritten later
TEAM_NAMES_INPUT = TEAM_NAMES

# Time the script
start_time = datetime.now()

for year in YEARS:
    print(f"------------------{year}------------------")
    adjust_team_names(team_abbreviations.pbp_abbrevs, year)
    adjust_team_names(team_abbreviations.boxscore_website_abbrevs, year)
    adjust_team_names(team_abbreviations.roster_website_abbrevs, year)
    if TEAM_NAMES_INPUT == "all":
        TEAM_NAMES = list(team_abbreviations.roster_website_abbrevs.keys())
        TEAM_NAMES.sort()

    # Process team records, game sites, and other game info for every game of
    # the year, for every team
    all_game_info_df = get_game_info(year)
    # Read play-by-play of ALL GAMES into pandas dataframe
    pbp_file = f"data/inputs/play_by_play/play_by_play_{year}.csv"
    game_df = pd.read_csv(pbp_file, low_memory=False)

    # Gather team roster for all teams, all weeks of input year
    all_rosters_df = process_rosters(year, WEEKS, ROSTER_FILTER_FILE)

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
                    full_team_name, game_df, roster_df, game_info_ser, GAME_TIMES
                )

                if TRACK_MIDGAME_STATS:
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
if TRACK_MIDGAME_STATS:
    all_stats_df = all_stats_df.reset_index().set_index(
        ["Player", "Year", "Week", "Elapsed Time"]
    )
    print(all_stats_df)

if SAVE_DATA:
    all_boxes_df.to_csv(OUTPUT_FILE_BOX)
    if TRACK_MIDGAME_STATS:
        all_stats_df.to_csv(OUTPUT_FILE_MIDGAME)

print(f"Elapsed Time: {(datetime.now() - start_time).total_seconds()} seconds")
