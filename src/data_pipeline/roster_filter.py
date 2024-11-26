import matplotlib.pyplot as plt
from misc.nn_helper_functions import stats_to_fantasy_points

# Processes boxscore stats and rosters to create a short list of players to focus data collection on
# Generates a csv of the players to hone in on

def generate_roster_filter(input_dfs, final_stats_df, save_file, num_players=300, plot_filter=False):

    # Get most recent roster data from input dfs
    rosters_df = input_dfs[-1][1].copy()

    # Add Fantasy Pointst to final stats if not already computed
    final_stats_df = stats_to_fantasy_points(final_stats_df.reset_index())

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
    fantasy_avgs = final_stats_df.loc[:, ['Player', 'Fantasy Points']].groupby(
        ['Player']).mean().rename(columns={'Fantasy Points': 'Fantasy Avg'})
    player_list_df = player_list_df.merge(
        right=fantasy_avgs,
        left_on='full_name',
        right_on='Player')

    # 3. Count games played - instances of player name
    game_counts = final_stats_df['Player'].value_counts()
    player_list_df = player_list_df[player_list_df['full_name'].apply(
        lambda x: game_counts[x] >= 5)]

    # 4a. Sort by max avg fantasy points
    player_list_df = player_list_df.sort_values(
        by=['Fantasy Avg'], ascending=False).reset_index()
    # 4b. Take first x (num_players) players
    player_list_df = player_list_df.iloc[0:num_players]

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
    player_list_df.to_csv(save_file)
    print(f'Saved Roster Filter to {save_file}')

    # Print data and breakdown by team/position
    print('================= Roster Filter ==================')
    print(player_list_df)
    print('================== Breakdown by Team =================')
    print(player_list_df['Team'].value_counts())
    print('================== Breakdown by Position =================')
    print(player_list_df['Position'].value_counts())

    if plot_filter:
        create_filter_plot(player_list_df, num_players)

    return player_list_df


def apply_roster_filter(midgame_df, final_stats_df, filter_df):
    filter_names = filter_df['Name'].to_list()
    midgame_df = midgame_df[midgame_df['Player'].apply(lambda x: x in filter_names)]
    final_stats_df = final_stats_df[final_stats_df.apply(lambda x: x.name in filter_names,axis=1)]

    return midgame_df, final_stats_df


def create_filter_plot(player_list_df, num_players):
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
    ax = plt.subplots(1, 1)[1]
    ax.bar(range(1, num_players + 1), player_list_df['Fantasy Avg'].to_list(),
        width=1, color=plot_colors, linewidth=0)
    plt.xlabel('Rank')
    plt.ylabel('Avg. Fantasy Score')
    plt.title('Fantasy Performance of Filtered Player List')

    def lp(i):
        return ax.plot([], color=position_colors[i], label=i)[0]

    leg_handles = [lp(i) for i in position_colors]
    plt.legend(handles=leg_handles)
    plt.show()
