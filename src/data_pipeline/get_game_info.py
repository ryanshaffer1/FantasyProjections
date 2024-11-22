import dateutil.parser as dateparse
import pandas as pd
import numpy as np
from data_pipeline import team_abbreviations

def get_game_info(year, pbp_df):
    # For a given year, generates info on every game for each team: who is home vs away, and records of each team going into the game
    # Also generates url to get game stats from pro-football-reference.com
    url_intro = 'https://www.pro-football-reference.com/boxscores/'

    # Filter df to only the final play of each game
    pbp_df = pbp_df.drop_duplicates(subset='game_id',keep='last')

    # Filter to only regular-season games
    pbp_df = pbp_df[pbp_df['season_type']=='REG']
    num_weeks = len(set(pbp_df['week']))

    # Output data frame
    all_games_df = pd.DataFrame()

    for full_team_name, team_abbr in team_abbreviations.pbp_abbrevs.items():
        # Filter to only games including the team of interest
        scores_df = pbp_df.copy()
        scores_df = scores_df[(scores_df['home_team'] == team_abbr)
                              | (scores_df['away_team'] == team_abbr)]

        # Track year (maybe unnecessary)
        scores_df['Year'] = year

        # Use full team name instead of abbreviation
        for col in ['home_team','away_team']:
            scores_df[col] = scores_df[col].apply(lambda x: team_abbreviations.invert(team_abbreviations.pbp_abbrevs)[x])

        # Track team name and opponent name
        scores_df['Team Name'] = full_team_name
        scores_df['Opp Name'] = scores_df.apply(lambda x:
            x['away_team'] if x['home_team'] == x['Team Name'] else x['home_team'],
            axis=1)

        # Track game site and home/away team names
        scores_df['Site'] = scores_df.apply(
            lambda x: 'Home' if x['home_team'] == x['Team Name'] else 'Away', axis=1)
        scores_df = scores_df.rename(columns={'week':'Week','home_team':'Home Team Name','away_team':'Away Team Name'})


        # URL to get stats from
        scores_df['game_date'] = scores_df.apply(lambda x:
            dateparse.parse(x['game_date']).strftime('%Y%m%d'), axis=1)
        scores_df['Stats URL'] = scores_df.apply(lambda x:
            url_intro + x['game_date'] + '0' +
            team_abbreviations.roster_website_abbrevs[x['Home Team Name']] +
            '.htm', axis=1)

        # Track ties, wins, and losses
        scores_df['Tie'] = scores_df['total_home_score'] == scores_df['total_away_score']
        scores_df['Win Loc'] = scores_df.apply(lambda x: 'Home' if x['total_home_score']>=x['total_away_score'] else 'Away', axis=1)
        scores_df['Win'] = (scores_df['Win Loc'] == scores_df['Site']) & (np.logical_not(scores_df['Tie']))
        scores_df['Loss'] = (scores_df['Win Loc'] != scores_df['Site']) & (np.logical_not(scores_df['Tie']))
        with pd.option_context("future.no_silent_downcasting", True):
            # 1s and 0s instead of True/False, so we can add them up next
            scores_df = scores_df.replace(to_replace=[True, False], value=[1, 0])

        # Track team record heading into each week
        game_info_df = scores_df.copy().set_index('Week').reset_index()

        # Get team record following each week
        game_info_df['Postgame Wins'] = game_info_df['Win'].cumsum()
        game_info_df['Postgame Losses'] = game_info_df['Loss'].cumsum()
        game_info_df['Postgame Ties'] = game_info_df['Tie'].cumsum()
        # Manipulate to instead show the team's record going into each week
        num_weeks_with_games = game_info_df.shape[0]
        game_info_df['Team Wins'] = [0] + game_info_df.loc[0:num_weeks_with_games - 2, 'Postgame Wins'].to_list()
        game_info_df['Team Losses'] = [0] + game_info_df.loc[0:num_weeks_with_games - 2, 'Postgame Losses'].to_list()
        game_info_df['Team Ties'] = [0] + game_info_df.loc[0:num_weeks_with_games - 2, 'Postgame Ties'].to_list()

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
            'Team Ties']
        game_info_df = game_info_df[columns].set_index(['Week', 'Team Name']).sort_index()

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
    all_games_df = all_games_df.reset_index().set_index(
        ['Team Name', 'Year', 'Week']).sort_index()
    # all_games_df = all_games_df.reset_index()[columns].set_index(
    #     ['Team Name', 'Year', 'Week']).sort_index()

    return all_games_df
