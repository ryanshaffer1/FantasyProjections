"""Contains variables defining the file names and folder structures used in obtaining and processing source data."""  # fmt: skip

# Folders
DATA_FOLDER = "data/"
INPUT_FOLDER = DATA_FOLDER + "inputs/"
OUTPUT_FOLDER = DATA_FOLDER + "stats/"
MISC_FOLDER = DATA_FOLDER + "misc/"
ODDS_FOLDER = DATA_FOLDER + "odds/"
PRE_PROCESS_FOLDER = DATA_FOLDER + "to_nn/"

# Files
OUTPUT_FILE_FINAL_STATS = OUTPUT_FOLDER + "statlines.csv"
OUTPUT_FILE_MIDGAME = OUTPUT_FOLDER + "pbp_stats.csv"
ROSTER_FILTER_FILE = INPUT_FOLDER + "Filtered Player List.csv"

# Input file locations
ONLINE_URL_NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/"
online_file_paths = {
    "pbp": ONLINE_URL_NFLVERSE + "pbp/play_by_play_{0}.csv",
    "roster": ONLINE_URL_NFLVERSE + "weekly_rosters/roster_weekly_{0}.csv",
}
local_file_paths = {
    "pbp": INPUT_FOLDER + "play_by_play/play_by_play_{0}.csv",
    "roster": INPUT_FOLDER + "rosters/roster_weekly_{0}.csv",
}
# Neural Net processed stats files
NN_STAT_FILES = {"midgame": "midgame_data_to_nn.csv", "final": "final_stats_to_nn.csv", "id": "data_ids.csv"}

# Miscellaneous:
# Feature config file
FEATURE_CONFIG_FILE = MISC_FOLDER + "stats_config.csv"
# Pro-Football-Reference.com URL intros
PFR_BOXSCORE_URL_INTRO = "https://www.pro-football-reference.com/boxscores/"
PFR_PLAYER_URL_INTRO = "https://www.pro-football-reference.com/players/"
# Box-Score Parsing Validation files
TRUE_STATS_FILE = MISC_FOLDER + "true_box_scores.csv"
PARSING_VALIDATION_FILE = MISC_FOLDER + "pbp_parsing_validation.csv"
# Player ID dictionaries/cross-mapping files
MASTER_PLAYER_ID_FILE = MISC_FOLDER + "player_ids.csv"
PFR_ID_FILENAME = MISC_FOLDER + "names_to_pfr_ids.json"
# Sleeper data
SLEEPER_PROJ_DICT_FILE = MISC_FOLDER + "sleeper_projections_dict.json"
# Odds data
ODDS_FILE = ODDS_FOLDER + "odds.csv"
# Private configuration data (e.g. API Keys)
ODDS_API_KEY_FILE = "FantasyProjections/config/private/odds_api_key.txt"
