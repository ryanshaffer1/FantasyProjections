"""Creates and exports helper functions commonly used when gathering Fantasy Projections data.

    Functions:
        calc_game_time_elapsed : Calculates game-time elapsed (in minutes) from 0 to 60 minutes, given a DataFrame/Series with columns for qtr and time.
        adjust_team_names : Changes NFL team names and abbreviations in internal dictionaries to account for the changing of NFL team names over the years in real life.
        swap_team_names : Changes the NFL team name used as a key in the dictionary if the name is not appropriate for the year.
        weeks_played_by_team : Returns all weeks in a year in which a team played a game, and all weeks containing at least one player with stats being tracked.
        cleanup_data : Performs final cleanup of gathered NFL stats data, including trimming unnecessary columns and setting/sorting indices.
"""
import dateutil.parser as dateparse
import numpy as np
import pandas as pd
from data_pipeline import team_abbreviations


def calc_game_time_elapsed(data):
    """Calculates game-time elapsed (in minutes) from 0 to 60 minutes, given a DataFrame/Series with columns for qtr and time.

        Args:
            data (pandas.DataFrame | pandas.Series): Contains one or multiple instances in time, including data labeled "qtr" and data labeled "time"
                where qtr denotes the current quarter and time denotes the time on the game clock (e.g. "15:00" down to "0:00").

        Returns:
            pandas.Series | float: Time elapsed since the start of the game, in minutes. If data is a DataFrame, return a Series; if data is a Series, return a float.
    """

    if isinstance(data,pd.Series):
        qtr = data["qtr"]
        time = data["time"]
    elif isinstance(data,pd.DataFrame):
        qtr = data.loc[:,"qtr"]
        time = data.loc[:,"time"]
    else:
        print('Must input data as pd.Series or pd.DataFrame')
        qtr = None
        time = None

    if isinstance(time, float):
        minutes = "15"
        seconds = "0"
    else:
        minutes = time.split(":")[0]
        seconds = time.split(":")[1]
    time_rem_in_qtr = int(minutes) + int(seconds) / 60
    time_elapsed = round((qtr - 1) * 15 + 15 - time_rem_in_qtr, 2)

    return time_elapsed


def adjust_team_names(dictionaries, year):
    """Changes NFL team names and abbreviations in internal dictionaries to account for the changing of NFL team names over the years in real life.

        Ex. in 2020: Oakland Raiders -> Las Vegas Raiders
            If year < 2019.5, must use "Oakland Raiders" as the name to find the correct stats/info from data sources.
            If year > 2019.5, must use "Las Vegas Raiders" as the name to find the correct stats/info from data sources.
            - In some cases the abbreviations also change: OAK became LVR, or LV, depending on the data source.

        Args:
            dictionaries (dict | list of dicts): Dictionary or list of dictionaries with keys listing out NFL team names. 
            year (int): Current year being processed.

        Returns:
            list of dicts: list of dictionaries where each dict has had keys adjusted for any NFL team name changes.
    """

    # Note: the team name changes must be included below in reverse
    # chronological order (most important for teams w/ multiple changes, e.g.
    # Washington)

    # Handle edge case of singular dict being input
    if isinstance(dictionaries,dict):
        dictionaries = [dictionaries]

    for dictionary in dictionaries:
        # 2022: Washington Football Team -> Washington Commanders
        dictionary = swap_team_names(
            year, dictionary, 2021.5, "Washington Football Team", "Washington Commanders"
        )

        # 2020: Washington Redskins -> Washington Football Team
        dictionary = swap_team_names(
            year, dictionary, 2019.5, "Washington Redskins", "Washington Football Team"
        )

        # 2020: Oakland Raiders -> Las Vegas Raiders
        dictionary = swap_team_names(
            year, dictionary, 2019.5, "Oakland Raiders", "Las Vegas Raiders"
        )
        # also need to make a value (abbreviation) swap for this one
        if "Oakland Raiders" in dictionary.keys() and dictionary["Oakland Raiders"] in ["LV","LVR"]:
            dictionary["Oakland Raiders"] = "OAK"
        if (
            "Las Vegas Raiders" in dictionary.keys()
            and dictionary["Las Vegas Raiders"] == "OAK"
        ):
            # Swapping from Oakland to Las Vegas. One of the dictionaries needs "LV", one needs "LVR".
            # This helper function doesn't natively know which dict it is working with.
            # Using a hack where we see what Kansas City has as its value, and make
            # the change accordingly...
            if dictionary["Kansas City Chiefs"] == "KC":
                dictionary["Las Vegas Raiders"] = "LV"
            else:
                dictionary["Las Vegas Raiders"] = "LVR"

    return dictionaries


def swap_team_names(year, dictionary, year_threshold, before_name, after_name):
    """Changes the NFL team name used as a key in the dictionary if the name is not appropriate for the year.

        Args:
            year (int): Current year being processed.
            dictionary (dict): Dictionary with keys listing out NFL team names.
            year_threshold (float): Year where name transition occurred. Should be xxxx.5 (e.g. 2021.5) so that "before" and "after" are unambiguous.
            before_name (str): Team Name prior to year_threshold
            after_name (str): Team Name after year_threshold

        Returns:
            dict: Dictionary with the team name changed if necessary.
    """

    if year < year_threshold:
        if after_name in dictionary.keys():
            dictionary[before_name] = dictionary[after_name]
            del dictionary[after_name]
    else:
        if before_name in dictionary.keys():
            dictionary[after_name] = dictionary[before_name]
            del dictionary[before_name]

    return dictionary


def weeks_played_by_team(all_game_info_df, all_rosters_df, full_team_name, year):
    """Returns all weeks in a year in which a team played a game, and all weeks containing at least one player with stats being tracked.

        Args:
            all_game_info_df (pandas.DataFrame): Contains information on all NFL games being processed, including team names, etc.
            all_rosters_df (pandas.DataFrame): Contains week-by-week NFL rosters listing out the players being tracked.
            full_team_name (str): Name of team being processed.
            year (int): Current year being processed.

        Returns:
            list: List of all regular season weeks where games have been played by the team (excludes byes, future weeks)
            list : List of all regular season weeks where a player in the roster DataFrame played.
                Helpful when roster DataFrame has been filtered to a subset of NFL players.
    """

    # All regular season weeks where games have been played by the team
    # (excludes byes, future weeks)
    weeks_with_games = list(all_game_info_df.loc[(full_team_name, year)].index)
    weeks_with_players = (
        all_rosters_df.loc[(team_abbreviations.pbp_abbrevs[full_team_name])]
        .index.unique()
        .to_list()
    )

    return weeks_with_games, weeks_with_players


def cleanup_data(midgame_df, final_stats_df, list_of_stats='default'):
    """Performs final cleanup of gathered NFL stats data, including trimming unnecessary columns and setting/sorting indices.

        Args:
            midgame_df (pandas.DataFrame): Stats accrued over the course of an NFL game for a set of players/games.
            final_stats_df (pandas.DataFrame): Stats at the end of an NFL game for a set of players/games.
            list_of_stats (list | str, optional): List of statistics to include in final stats data. Defaults to 'default'.
            
        Returns:
            pandas.DataFrame: midgame stats DataFrame, cleaned.
            pandas.DataFrame: final stats DataFrame, cleaned.
    """

    # Default stat list
    if list_of_stats == 'default':
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
        ]

    # Organize dfs
    midgame_df = midgame_df.reset_index().set_index(
        ["Player", "Year", "Week", "Elapsed Time"]
    )

    final_stats_df = (
        final_stats_df.reset_index().set_index(["Player", "Year", "Week"]).sort_index()
    )

    final_stats_df = final_stats_df[["Team", "Opponent", "Position", "Age", "Site"] + list_of_stats]

    return midgame_df, final_stats_df


def compute_team_record(scores_df):
    """Computes the record (wins, losses, and ties) of a team GOING INTO each game (i.e. not including the result of the current game).

        Args:
            scores_df (pandas.DataFrame): DataFrame with a row for each game, including the score of each game.
                Must contain all games from Week 1 through the current (or final) week, with no skipped games, for each team being handled (may be one team at a time or the entire league at once).

        Returns:
            pandas.DataFrame: Input DataFrame, with three columns added: "Team Wins", "Team Losses", "Team Ties". These correspond to the team's record GOING INTO the game.
    """

    # Add columns tracking whether each game is a win, loss, or tie for the team
    scores_df = track_wins_losses_ties(scores_df)

    # Track team record heading into each week
    scores_df = scores_df.sort_values(by='Week')

    # Get team record following each week
    scores_df['Postgame Wins'] = scores_df.groupby('Team Abbrev')['Win'].transform(pd.Series.cumsum)
    scores_df['Postgame Losses'] = scores_df.groupby('Team Abbrev')['Loss'].transform(pd.Series.cumsum)
    scores_df['Postgame Ties'] = scores_df.groupby('Team Abbrev')['Tie'].transform(pd.Series.cumsum)

    # Manipulate to instead show the team's record going into each week
    games_played_by_team = scores_df.groupby('Team Abbrev')['Week'].nunique()
    scores_df = scores_df.set_index(['Team Abbrev','Week']).sort_index()
    for new_col, old_col in zip(['Team Wins', 'Team Losses', 'Team Ties'],['Postgame Wins', 'Postgame Losses', 'Postgame Ties']):
        scores_df[new_col] = shift_val_one_game_back(scores_df[old_col].to_list(),games_played_by_team)
    scores_df = scores_df.reset_index()

    return scores_df


def track_wins_losses_ties(scores_df):
    """Tracks whether each game is a win, loss, or tie for the team.

        Args:
            scores_df (pandas.DataFrame): DataFrame with a row for each game, including the score of each game.
        
        Returns:
            pandas.DataFrame: Input DataFrame, with three columns added: "Win", "Loss", "Tie". These correspond to the team's record GOING INTO the game.
    """

    scores_df['Tie'] = scores_df['total_home_score'] == scores_df['total_away_score']
    scores_df['Win Loc'] = scores_df.apply(lambda x: 'Home' if x['total_home_score']>=x['total_away_score'] else 'Away', axis=1)
    scores_df['Win'] = (scores_df['Win Loc'] == scores_df['Site']) & (np.logical_not(scores_df['Tie']))
    scores_df['Loss'] = (scores_df['Win Loc'] != scores_df['Site']) & (np.logical_not(scores_df['Tie']))
    with pd.option_context("future.no_silent_downcasting", True):
        # 1s and 0s instead of True/False, so we can add them up next
        scores_df = scores_df.replace(to_replace=[True, False], value=[1, 0])

    return scores_df


def shift_val_one_game_back(postgame_vals_by_team_week, games_played_by_team):
    """Converts a team's wins, losses, or ties over the course of a season from post-game to pre-game values.

        Can process multiple teams simultaneously, or one game at a time (current use case).

        Args:
            postgame_vals_by_team_week (list): List of postgame wins OR losses OR ties for all weeks/teams currently being processed
            games_played_by_team (pandas.Series): Number of games played by each team currently being processed

        Returns:
            list: List of pregame wins OR losses OR ties for all weeks/teams currently being processed
    """

    last_game_played_by_team = [x-1 for x in games_played_by_team.cumsum().to_list()]
    last_game_played_by_team.reverse()
    # Remove last games played by each team
    for ind in last_game_played_by_team:
        del postgame_vals_by_team_week[ind]
    # Add a zero at the beginning of each team's season
    zero_indices = [0] + games_played_by_team.cumsum().iloc[0:-1].to_list()
    for ind in zero_indices:
        postgame_vals_by_team_week[ind:ind] = [0]

    return postgame_vals_by_team_week


def parse_year_from_date(x):
    """Parses year from input date string.

        If parsing fails, returns 2000 as the year.

        Args:
            x (str): Date, in flexible date string format

        Returns:
            int: Year from date
    """

    try:
        output = dateparse.parse(x).year
    except TypeError:
        output = 2000 # whatever man
    return output


def clean_team_names(team_names, year):
    # Processing only needed if team_names=='all'
    if team_names == 'all':

        # Adjust team names and abbreviations for year (team name changes over the years)
        adjust_team_names((team_abbreviations.pbp_abbrevs,
                        team_abbreviations.boxscore_website_abbrevs,
                        team_abbreviations.roster_website_abbrevs),
                        year)
        # Take all team names from dict
        team_names = list(team_abbreviations.roster_website_abbrevs.keys())

    return team_names


def gen_game_play_by_play(pbp_df, game_id):

    # Make a copy of the input play-by-play df
    pbp_df = pbp_df.copy()

    # Filter to only the game of interest (using game_id)
    pbp_df = pbp_df[pbp_df['game_id'] == game_id]

    # Elapsed time
    pbp_df.loc[:,'Elapsed Time'] = pbp_df.apply(calc_game_time_elapsed, axis=1)

    # Sort by ascending elapsed time
    pbp_df = pbp_df.set_index('Elapsed Time').sort_index(ascending=True)

    return pbp_df


def filter_game_time(player_stats_df, game_times):
    """Reduces the size of the midgame stats data by sampling the data at discrete game times, rather than keeping the stats after every single play.
    
        Rounds the game time at the start of each play to the nearest value in game_times, then keeps the first instance of each value, as well as the final row (to capture final stats).
        Note that there may be an additional (redundant) row for the final stat line.

        Args:
            player_stats_df (pandas.DataFrame): Player's accumulated stat line after every single play in the game.
            game_times (list | str): List of game times to include in the output. If string is input (such as "all"), no filtering occurs.

        Returns:
            pandas.DataFrame: Player's accumulated stat line at each designated game time. May have an additional (redundant) row for the final stat line.
    """

    if not isinstance(game_times, str):
        player_stats_df['Rounded Time'] = player_stats_df.index.map(
            lambda x: game_times[abs(game_times - float(x)).argmin()])
        player_stats_df = pd.concat((player_stats_df.iloc[1:].drop_duplicates(
            subset='Rounded Time', keep='first'),player_stats_df.iloc[-1].to_frame().T))
        player_stats_df = player_stats_df.reset_index(drop=True).rename(
            columns={'Rounded Time': 'Elapsed Time'}).set_index('Elapsed Time')

    return player_stats_df
