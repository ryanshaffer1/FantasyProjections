from datetime import datetime
import pandas as pd
import numpy as np
import team_abbreviations

def get_game_info(year):
    # For a given year, generates info on every game for each team: who is home vs away, and records of each team going into the game
    # Also generates url to get game stats from pro-football-reference.com

    # Source url for boxscore stats
    url_intro = 'https://www.pro-football-reference.com/boxscores/'

    # All teams
    team_names = team_abbreviations.roster_website_abbrevs.keys()  # All team names

    # Read spreadsheet with all game scores over the year
    gamescores_file = f'data/inputs/gamescores/gamescores_{year}.csv'
    all_scores_df = pd.read_csv(gamescores_file, dtype={'Week': str})

    # Filter to only completed, regular-season games - will have the term
    # "boxscore" instead of "preview", and a numeric week number (e.g. not
    # "WildCard")
    all_scores_df = all_scores_df[(
        all_scores_df['Unnamed: 7'] == 'boxscore') & all_scores_df['Week'].str.isnumeric()]
    all_scores_df['Week'] = [int(i) for i in all_scores_df['Week'].to_list()]
    num_weeks = len(set(all_scores_df['Week']))

    # Output data frame
    all_games_df = pd.DataFrame()

    for full_team_name in team_names:
        # Filter to only games including the team of interest
        scores_df = all_scores_df.copy()
        scores_df = scores_df[(scores_df['Winner/tie'] == full_team_name)
                              | (scores_df['Loser/tie'] == full_team_name)]

        # Track year (maybe unnecessary)
        scores_df['Year'] = year

        # Track team name and opponent name
        scores_df['Team Name'] = full_team_name
        scores_df['Opp Name'] = scores_df.apply(lambda x:
            x['Loser/tie'] if x['Winner/tie'] == x['Team Name'] else x['Winner/tie'],
            axis=1)

        # Track game site and home/away team names
        scores_df['Home'] = ((scores_df['Loser/tie'] == full_team_name) & (scores_df['Unnamed: 5'] == '@')) | (
            (scores_df['Winner/tie'] == full_team_name) & (scores_df['Unnamed: 5'].isna()))
        scores_df['Site'] = scores_df.apply(
            lambda x: 'Home' if x['Home'] else 'Away', axis=1)
        scores_df['Home Team Name'] = scores_df.apply(
            lambda x: x['Team Name'] if x['Home'] else x['Opp Name'], axis=1)
        scores_df['Away Team Name'] = scores_df.apply(
            lambda x: x['Team Name'] if not x['Home'] else x['Opp Name'], axis=1)

        # URL to get stats from
        scores_df['Date'] = scores_df.apply(lambda x: datetime.strftime(
            datetime.strptime(x['Date'], '%m/%d/%Y'), '%Y%m%d'), axis=1)
        scores_df['Stats URL'] = scores_df.apply(lambda x:
            url_intro + x['Date'] + '0' +
            team_abbreviations.roster_website_abbrevs[x['Home Team Name']] +
            '.htm', axis=1)

        # Track ties, wins, and losses
        scores_df['Tie'] = scores_df['PtsW'] == scores_df['PtsL']
        scores_df['Win'] = (
            (scores_df['Winner/tie'] == full_team_name) & (np.logical_not(scores_df['Tie'])))
        scores_df['Loss'] = (
            (scores_df['Loser/tie'] == full_team_name) & (np.logical_not(scores_df['Tie'])))
        with pd.option_context("future.no_silent_downcasting", True):
            # 1s and 0s instead of True/False, so we can add them up next
            scores_df = scores_df.replace(to_replace=[True, False], value=[1, 0])

        # Track team record heading into each week
        game_info_df = scores_df.copy().set_index('Week').reset_index()

        game_info_df.drop(['Unnamed: 5',
                           'Unnamed: 7',
                           'PtsW',
                           'PtsL',
                           'YdsW',
                           'TOW',
                           'YdsL',
                           'TOL'],
                          axis=1,
                          inplace=True)

        # Get team record following each week
        game_info_df['Postgame Wins'] = game_info_df['Win'].cumsum()
        game_info_df['Postgame Losses'] = game_info_df['Loss'].cumsum()
        game_info_df['Postgame Ties'] = game_info_df['Tie'].cumsum()
        # Manipulate to instead show the team's record going into each week
        num_weeks_with_games = game_info_df.shape[0]
        game_info_df['Team Wins'] = [0] + game_info_df.loc[0:num_weeks_with_games - 2, 'Postgame Wins'].to_list()
        game_info_df['Team Losses'] = [0] + game_info_df.loc[0:num_weeks_with_games - 2, 'Postgame Losses'].to_list()
        game_info_df['Team Ties'] = [0] + game_info_df.loc[0:num_weeks_with_games - 2, 'Postgame Ties'].to_list()

        game_info_df = game_info_df.set_index(['Week', 'Team Name']).sort_index()

        # Append to dataframe of all teams' games
        all_games_df = pd.concat([all_games_df, game_info_df])

    # Add opponent's record to each game in the df
    for week in range(1, num_weeks + 1):
        all_games_df.loc[week, 'Opp Wins'] = all_games_df.loc[(
            week, all_games_df.loc[week, 'Opp Name']), 'Team Wins'].to_list()
        all_games_df.loc[week, 'Opp Losses'] = all_games_df.loc[(
            week, all_games_df.loc[week, 'Opp Name']), 'Team Losses'].to_list()
        all_games_df.loc[week, 'Opp Ties'] = all_games_df.loc[(
            week, all_games_df.loc[week, 'Opp Name']), 'Team Ties'].to_list()

    # Clean up df for output
    columns = [
        'Year',
        'Week',
        'Team Name',
        'Opp Name',
        'Stats URL',
        'Site',
        'Home Team Name',
        'Away Team Name',
        'Team Wins',
        'Team Losses',
        'Team Ties',
        'Opp Wins',
        'Opp Losses',
        'Opp Ties']
    all_games_df = all_games_df.reset_index()[columns].set_index(
        ['Team Name', 'Year', 'Week']).sort_index()

    return all_games_df
