import dateutil.parser as dateparse
import pandas as pd
import numpy as np
from data_pipeline import team_abbreviations

URL_INTRO = 'https://www.pro-football-reference.com/boxscores/'

def get_game_info(year, pbp_df):
    # For a given year, generates info on every game for each team: who is home vs away, and records of each team going into the game
    # Also generates url to get game stats from pro-football-reference.com

    # Filter df to only the final play of each game
    pbp_df = pbp_df.drop_duplicates(subset='game_id',keep='last')

    # Filter to only regular-season games
    pbp_df = pbp_df[pbp_df['season_type']=='REG']
    num_weeks = len(set(pbp_df['week']))

    # Output data frame
    all_games_df = pd.DataFrame()

    for team_abbr in team_abbreviations.pbp_abbrevs.values():
        # Filter to only games including the team of interest
        scores_df = pbp_df.copy()
        scores_df = scores_df[(scores_df['home_team'] == team_abbr)
                              | (scores_df['away_team'] == team_abbr)]

        # Track team name and opponent name
        scores_df['Team Abbrev'] = team_abbr
        scores_df['Opp Abbrev'] = scores_df.apply(lambda x:
            x['away_team'] if x['home_team'] == x['Team Abbrev'] else x['home_team'],
            axis=1)

        # Track game site and home/away team
        scores_df['Site'] = scores_df.apply(
            lambda x: 'Home' if x['home_team'] == x['Team Abbrev'] else 'Away', axis=1)
        scores_df = scores_df.rename(columns={'week':'Week','home_team':'Home Team Abbrev','away_team':'Away Team Abbrev'})

        # URL to get stats from
        scores_df['game_date'] = scores_df.apply(lambda x:
            dateparse.parse(x['game_date']).strftime('%Y%m%d'), axis=1)
        scores_df['Stats URL'] = scores_df.apply(lambda x:
            URL_INTRO + x['game_date'] + '0' +
            team_abbreviations.convert_abbrev(x['Home Team Abbrev'],
            team_abbreviations.pbp_abbrevs,team_abbreviations.roster_website_abbrevs
            ) + '.htm', axis=1)

        # Track ties, wins, and losses
        scores_df = compute_team_record(scores_df)

        # Remove unnecessary columns
        columns_to_keep = [
            'Week',
            'Team Abbrev',
            'Opp Abbrev',
            'Stats URL',
            'Site',
            'Home Team Abbrev',
            'Away Team Abbrev',
            'Team Wins',
            'Team Losses',
            'Team Ties']
        scores_df = scores_df[columns_to_keep].set_index(['Week', 'Team Abbrev']).sort_index()

        # Append to dataframe of all teams' games
        all_games_df = pd.concat([all_games_df, scores_df])

    # Add opponent's record to each game in the df
    for week in range(1, num_weeks + 1):
        all_games_df.loc[week, 'Opp Wins'] = all_games_df.loc[(
            week, all_games_df.loc[week, 'Opp Abbrev']), 'Team Wins'].to_list()
        all_games_df.loc[week, 'Opp Losses'] = all_games_df.loc[(
            week, all_games_df.loc[week, 'Opp Abbrev']), 'Team Losses'].to_list()
        all_games_df.loc[week, 'Opp Ties'] = all_games_df.loc[(
            week, all_games_df.loc[week, 'Opp Abbrev']), 'Team Ties'].to_list()

    # Add year to dataframe
    all_games_df['Year'] = year

    # Clean up df for output
    all_games_df = all_games_df.reset_index()
    all_games_df['Team Name'] = all_games_df['Team Abbrev'].apply(lambda x:
        team_abbreviations.invert(team_abbreviations.pbp_abbrevs)[x])
    all_games_df = all_games_df.set_index(
        ['Team Name', 'Year', 'Week']).sort_index()

    return all_games_df


def pbp_abbrev_to_roster_site_abbrev(pbp_abbrev):
    team_name = team_abbreviations.invert(team_abbreviations.pbp_abbrevs)[pbp_abbrev]
    roster_abbrev = team_abbreviations.roster_website_abbrevs[team_name]
    return roster_abbrev


def compute_team_record(scores_df):
    # Add columns tracking whether each game is a win, loss, or tie for the team
    scores_df = add_wins_losses_ties(scores_df)

    # Track team record heading into each week
    scores_df = scores_df.sort_values(by='Week')

    # Get team record following each week
    scores_df['Postgame Wins'] = scores_df.groupby('Team Abbrev')['Win'].transform(pd.Series.cumsum)
    scores_df['Postgame Losses'] = scores_df.groupby('Team Abbrev')['Loss'].transform(pd.Series.cumsum)
    scores_df['Postgame Ties'] = scores_df.groupby('Team Abbrev')['Tie'].transform(pd.Series.cumsum)

    # Manipulate to instead show the team's record going into each week
    games_played_by_team = scores_df.groupby('Team Abbrev')['Week'].nunique()
    scores_df = scores_df.set_index(['Team Abbrev','Week']).sort_index()
    scores_df['Team Wins'] = shift_val_one_game_back(scores_df['Postgame Wins'].to_list(),games_played_by_team)
    scores_df['Team Losses'] = shift_val_one_game_back(scores_df['Postgame Losses'].to_list(),games_played_by_team)
    scores_df['Team Ties'] = shift_val_one_game_back(scores_df['Postgame Ties'].to_list(),games_played_by_team)
    scores_df = scores_df.reset_index()

    return scores_df


def add_wins_losses_ties(scores_df):
    scores_df['Tie'] = scores_df['total_home_score'] == scores_df['total_away_score']
    scores_df['Win Loc'] = scores_df.apply(lambda x: 'Home' if x['total_home_score']>=x['total_away_score'] else 'Away', axis=1)
    scores_df['Win'] = (scores_df['Win Loc'] == scores_df['Site']) & (np.logical_not(scores_df['Tie']))
    scores_df['Loss'] = (scores_df['Win Loc'] != scores_df['Site']) & (np.logical_not(scores_df['Tie']))
    with pd.option_context("future.no_silent_downcasting", True):
        # 1s and 0s instead of True/False, so we can add them up next
        scores_df = scores_df.replace(to_replace=[True, False], value=[1, 0])

    return scores_df


def shift_val_one_game_back(list_of_val_by_team,games_played_by_team):
    last_game_played_by_team = [x-1 for x in games_played_by_team.cumsum().to_list()]
    last_game_played_by_team.reverse()
    # Remove last games played by each team
    for ind in last_game_played_by_team:
        del list_of_val_by_team[ind]
    # Add a zero at the beginning of each team's season
    zero_indices = [0] + games_played_by_team.cumsum().iloc[0:-1].to_list()
    for ind in zero_indices:
        list_of_val_by_team[ind:ind] = [0]

    return list_of_val_by_team
