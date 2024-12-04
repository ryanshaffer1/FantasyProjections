"""Creates and exports helper functions commonly used when gathering Fantasy Projections data.

    Functions:
        calc_game_time_elapsed : Calculates game-time elapsed (in minutes) from 0 to 60 minutes, given a DataFrame/Series with columns for qtr and time.
        adjust_team_names : Changes NFL team names and abbreviations in internal dictionaries to account for the changing of NFL team names over the years in real life.
        swap_team_names : Changes the NFL team name used as a key in the dictionary if the name is not appropriate for the year.
        weeks_played_by_team : Returns all weeks in a year in which a team played a game, and all weeks containing at least one player with stats being tracked.
        cleanup_data : Performs final cleanup of gathered NFL stats data, including trimming unnecessary columns and setting/sorting indices.
"""

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

    final_stats_df = final_stats_df[["Team", "Opponent", "Position", "Age"] + list_of_stats]

    return midgame_df, final_stats_df
