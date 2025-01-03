"""Creates and exports functions to generate and apply filtered lists of NFL players based on maximum average Fantasy Points per game.

    Functions:
        generate_roster_filter : Generates a short list of players to focus data collection on, based on highest average Fantasy Points per game.
        apply_roster_filter : Trims previously-generated NFL stats DataFrames (midgame and final stats) to only include players in a filtered list.
        create_filter_plot : Plots bar chart of average Fantasy Points for each player in filtered list (sorted in descending order), colored by position.
"""

import logging
import matplotlib.pyplot as plt
from misc.stat_utils import stats_to_fantasy_points

# Set up logger
logger = logging.getLogger('log')

def generate_roster_filter(rosters_df, final_stats_df, save_file=None, num_players=300, plot_filter=False):
    """Generates a short list of players to focus data collection on, based on highest average Fantasy Points per game.

        Rules
        1. "Currently" active players only (active at some point in the last season being processed)
        2. Sort by fantasy points per game played
        3. Player must have played in at least 5 games
        4. Take the top num_players number of players per criteria 2

        Saves filtered list of players to a csv file to be used for later data collection.

        Args:
            rosters_df (pandas.DataFrame): DataFrame containing all weekly NFL rosters for a given series. Loaded from nfl-verse
            final_stats_df (pandas.DataFrame): Stats for every player at the end of each game in the input_dfs DataFrames.
            save_file (str, optional): csv file path to save filtered player list. Defaults to None (file is not saved).
            num_players (int, optional): Number of players to include in list. Defaults to 300.
            plot_filter (bool, optional): Whether to visualize the filtered player list by position and average Fantasy Points. Defaults to False.

        Returns:
            pandas.DataFrame: List of players to include in the filter, along with some additional data like team, position, average Fantasy Points, etc.
    """

    # Add Fantasy Pointst to final stats if not already computed
    final_stats_df = stats_to_fantasy_points(final_stats_df.reset_index())

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
    if save_file:
        player_list_df.to_csv(save_file)
        logger.info(f'Saved Roster Filter to {save_file}')

    # Print data and breakdown by team/position
    logger.info('Roster Filter Breakdown by Team:')
    logger.info(f'{player_list_df['Team'].value_counts()}')
    logger.info('Roster Filter Breakdown by Position:')
    logger.info(f'{player_list_df['Position'].value_counts()}')

    if plot_filter:
        create_filter_plot(player_list_df, num_players)

    return player_list_df


def apply_roster_filter(midgame_df, final_stats_df, filter_df):
    """Trims previously-generated NFL stats DataFrames (midgame and final stats) to only include players in a filtered list.

        Args:
            midgame_df (pandas.DataFrame): Stats accrued over the course of an NFL game, over a range of players/games.
            final_stats_df (pandas.DataFrame): Stats at the end of an NFL game, over a range of players/games.
            filter_df (pandas.DataFrame): List of players to include in the filtered data output.

        Returns:
            pandas.DataFrame: midgame_df, trimmed to only include the players in filter_df.
            pandas.DataFrame: final_stats_df, trimmed to only include the players in filter_df.
    """

    filter_names = filter_df['Name'].to_list()
    midgame_df = midgame_df[midgame_df['Player'].apply(lambda x: x in filter_names)]
    final_stats_df = final_stats_df[final_stats_df.apply(lambda x: x.name in filter_names,axis=1)]

    return midgame_df, final_stats_df


def create_filter_plot(player_list_df, num_players):
    """Plots bar chart of average Fantasy Points for each player in filtered list (sorted in descending order), colored by position.

        Args:
            player_list_df (pandas.DataFrame): List of players included in the filter, along with some additional data like team, position, average Fantasy Points, etc.
            num_players (int): Number of players to include in list.
    """

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
