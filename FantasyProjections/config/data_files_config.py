"""Contains variables defining the file names and folder structures used in obtaining and processing source data.
"""

# Folders
DATA_FOLDER = 'data2/'
INPUT_FOLDER = DATA_FOLDER + 'inputs/'
OUTPUT_FOLDER = DATA_FOLDER + 'stats/'
PRE_PROCESS_FOLDER = DATA_FOLDER + 'to_nn/'

# Files
OUTPUT_FILE_FINAL_STATS = OUTPUT_FOLDER + 'statlines.csv'
OUTPUT_FILE_MIDGAME = OUTPUT_FOLDER + 'pbp_stats.csv'
ROSTER_FILTER_FILE = INPUT_FOLDER + 'Filtered Player List.csv'

# Input file locations
ONLINE_DATA_SOURCE = 'https://github.com/nflverse/nflverse-data/releases/download/'
online_file_paths = {'pbp': ONLINE_DATA_SOURCE + 'pbp/play_by_play_{0}.csv',
                     'roster': ONLINE_DATA_SOURCE + 'weekly_rosters/roster_weekly_{0}.csv'}
local_file_paths = {'pbp':INPUT_FOLDER + 'play_by_play/play_by_play_{0}.csv',
                    'roster': INPUT_FOLDER + 'rosters/roster_weekly_{0}.csv'}

# Additional info sources
URL_INTRO = 'https://www.pro-football-reference.com/boxscores/'
