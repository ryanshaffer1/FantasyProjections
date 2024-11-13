import pandas as pd
import matplotlib.pyplot as plt

# Processes boxscore stats and rosters to create a short list of players to focus data collection on
# Generates a csv of the players to hone in on

# Inputs
NUM_PLAYERS = 300  # Number of players to use
SAVE_DATA = True
PLOT_DATA = True
SAVE_FILE = 'data/input data/Filtered Player List.csv'
BOXSCORES_FILE = 'data/statlines_allPlayers.csv'
CURRENT_ROSTER_FILE = 'data/input data/rosters/roster_weekly_2024.csv'

# Read files
boxscores_df = pd.read_csv(BOXSCORES_FILE)
rosters_df = pd.read_csv(CURRENT_ROSTER_FILE)

# Rules
# 1. Currently active players only (active at some point this season)
# 2. Sort by fantasy points per game played
# 3. Player must have played in at least 5 games
# 4. Take the top X number of players per criteria 2

# 1. Active players only
# Removes 'RET','CUT','DEV', 'TRC' (I think means free agent?)
active_statuses = ['ACT', 'INA', 'RES', 'EXE']
rosters_df = rosters_df[rosters_df.apply(
    lambda x: x['status'] in active_statuses, axis=1)]
# 1b. Remove tracking of week-by-week status - only one row per player.
# Take the entry from the last week they've played
last_week_played = rosters_df.loc[:, [
    'full_name', 'week']].groupby(['full_name']).max()
player_list_df = rosters_df[rosters_df.apply(lambda x: (x['full_name'] in last_week_played.index.to_list()) & (
    x['week'] == last_week_played.loc[x['full_name'], 'week']), axis=1)]

# 2. Add average fantasy points per game to df
fantasy_avgs = boxscores_df.loc[:, ['Player', 'Fantasy Score']].groupby(
    ['Player']).mean().rename(columns={'Fantasy Score': 'Fantasy Avg'})
player_list_df = player_list_df.merge(
    right=fantasy_avgs,
    left_on='full_name',
    right_on='Player')

# 3. Count games played - instances of player name
game_counts = boxscores_df['Player'].value_counts()
print(player_list_df)
player_list_df = player_list_df[player_list_df['full_name'].apply(
    lambda x: game_counts[x] >= 5)]
print(player_list_df)

# 4a. Sort by max avg fantasy points
player_list_df = player_list_df.sort_values(
    by=['Fantasy Avg'], ascending=False).reset_index()
# 4b. Take first x (num_players) players
player_list_df = player_list_df.iloc[0:NUM_PLAYERS]

# Clean up df for saving
player_list_df = player_list_df[['full_name',
                                 'gsis_id',
                                 'Fantasy Avg',
                                 'team',
                                 'position',
                                 'jersey_number']]
player_list_df = player_list_df.rename(
    columns={
        'team': 'Team',
        'position': 'Position',
        'jersey_number': 'Number',
        'full_name': 'Name',
        'gsis_id': 'Player ID'})

# Save
if SAVE_DATA:
    player_list_df.to_csv(SAVE_FILE)

# Print data and breakdown by team/position
print('================= Output Data ==================')
print(player_list_df)
print('================== Breakdown by Team =================')
print(player_list_df['Team'].value_counts())
print('================== Breakdown by Position =================')
print(player_list_df['Position'].value_counts())

# Plot bar chart of points by rank, colored by position
position_colors = {
    'QB': 'tab:blue',
    'RB': 'tab:orange',
    'WR': 'tab:green',
    'TE': 'tab:purple',
    'Other': 'tab:brown'}
plot_colors = player_list_df['Position'].apply(
    lambda x: position_colors[x] if x in position_colors else position_colors['Other'])
if position_colors['Other'] not in plot_colors:
    del position_colors['Other']
fig, ax = plt.subplots(1, 1)
ax.bar(range(1, NUM_PLAYERS + 1), player_list_df['Fantasy Avg'].to_list(),
       width=1, color=plot_colors, linewidth=0)
plt.xlabel('Rank')
plt.ylabel('Avg. Fantasy Score')
plt.title('Fantasy Performance of Filtered Player List')


def lp(i):
    return ax.plot([], color=position_colors[i], label=i)[0]


leg_handles = [lp(i) for i in position_colors]
plt.legend(handles=leg_handles)
plt.show()
