"""Validates that data used in Fantasy Football stat prediction matches statistics gathered from an independent source. 
    
    This module is a script to be run alone. Before running, the packages in requirements.txt must be installed. Use the following terminal command:
    > pip install -u requirements.txt
"""

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from scrape_pro_football_reference import scrape_box_score
from data_helper_functions import adjust_team_names
from get_game_info import get_game_info
import team_abbreviations

# Inputs
TRUE_STATS_FILE = 'data/true_box_scores.csv'
PBP_STATS_FILE = 'data/statlines.csv'
SAVE_DATA = False
SCRAPE_STATS = False

TEAM_NAMES = 'all'  # All team names
YEARS = range(2020, 2025)
WEEKS = range(1, 19)

# Dummy variable for a variable that might be overwritten later
TEAM_NAMES_INPUT = TEAM_NAMES

# Time the script
startTime = datetime.now()

# Scrape "true" box score stats from ProFootballReference
if SCRAPE_STATS:
    true_df = pd.DataFrame()
    for year in YEARS:
        print(f'------------------{year}------------------')
        adjust_team_names(team_abbreviations.pbp_abbrevs, year)
        adjust_team_names(team_abbreviations.boxscore_website_abbrevs, year)
        adjust_team_names(team_abbreviations.roster_website_abbrevs, year)
        if TEAM_NAMES_INPUT == 'all':
            TEAM_NAMES = list(team_abbreviations.roster_website_abbrevs.keys())
            TEAM_NAMES.sort()

        all_game_info_df = get_game_info(year)
        for full_team_name in TEAM_NAMES:
            print(f'---------{full_team_name}---------')
            # All regular season weeks where games have been played by the team
            # (excludes byes, future weeks)
            weeks_with_games = list(
                all_game_info_df.loc[(full_team_name, year)].index)
            for week in WEEKS:
                if week not in weeks_with_games:
                    print(f'Week {week} - no game found')
                else:
                    print(f'Week {week}')
                    # URL to pull stats for specific team & week
                    stats_html = all_game_info_df.loc[(
                        full_team_name, year, week), 'Stats URL']

                    # Grab box score (final game) stats from website
                    boxscore = scrape_box_score(stats_html, full_team_name)
                    boxscore['Week'] = week
                    boxscore['Year'] = year
                    # Add to running list of weeks
                    true_df = pd.concat([true_df, boxscore])

    # Format and save
    true_df = true_df.reset_index().set_index(
        ['Player', 'Year', 'Week']).sort_index()
    true_df.to_csv(TRUE_STATS_FILE)

else:
    true_df = pd.read_csv(TRUE_STATS_FILE)

# Load statlines gathered from play-by-play
pbp_df = pd.read_csv(PBP_STATS_FILE)

# Merge dataframes and compute differences in statistics
merged_df = pd.merge(
    true_df.reset_index(), pbp_df.reset_index(), how='inner', on=[
        'Player', 'Year', 'Week'])
stats = [
    'Pass Att',
    'Pass Cmp',
    'Pass Yds',
    'Pass TD',
    'Int',
    'Rush Att',
    'Rush Yds',
    'Rush TD',
    'Rec', 
    'Rec Yds',
    'Rec TD',
    'Fmb',
    'Fantasy Score']
for stat in stats:
    merged_df[f'{stat}_diff'] = merged_df[f'{stat}_y'] - \
        merged_df[f'{stat}_x']  # Estimated minus truth
diff_df = merged_df.copy()
diff_df = diff_df[['Player', 'Year', 'Week'] + [s + '_diff' for s in stats]]

if SAVE_DATA:
    diff_df.to_csv('data/misc/pbp_parsing_validation.csv')

print(f'Elapsed Time: {(datetime.now() - startTime).total_seconds()} seconds')

# Let's make a scatter plot
x = []
for j in range(diff_df.shape[0]):
    for i in range(len(stats)):
        x.append(i)
y = diff_df[[s + '_diff' for s in stats]].stack()

fig, ax = plt.subplots(1, 1)
ax.scatter(x, y)
ax.set_xticks(range(len(stats)))
ax.set_xticklabels(stats)
for label in ax.get_xticklabels():
    label.set(rotation=45, horizontalalignment='right')
plt.show()
