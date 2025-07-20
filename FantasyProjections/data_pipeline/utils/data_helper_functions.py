"""Creates and exports helper functions commonly used when gathering Fantasy Projections data.

    Functions:
        clean_team_names : Modifies list of team names to process data for. Primarily used to generate a list of team names to replace the "all" placeholder.
        compute_team_record : Computes the record (wins, losses, and ties) of a team GOING INTO each game (i.e. not including the result of the current game).
        construct_game_id : Generates Game ID as used by nfl-verse from a set of game information.
        single_game_play_by_play : Filters and cleans play-by-play data for a specific game; keeps all plays from that game and sorts by increasing elapsed game time.
        subsample_game_time : Reduces the size of the midgame stats data by sampling the data at discrete game times, rather than keeping the stats after every single play.
"""  # fmt: skip

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from data_pipeline.utils import team_abbreviations as team_abbrs

# Set up logger
logger = logging.getLogger("log")


def calc_game_time_elapsed(data):
    """Calculates game-time elapsed (in minutes) from 0 to 60 minutes, given a DataFrame/Series with columns for qtr and time.

        Args:
            data (pandas.DataFrame | pandas.Series): Contains one or multiple instances in time, including data labeled "qtr" and data labeled "time"
                where qtr denotes the current quarter and time denotes the time on the game clock (e.g. "15:00" down to "0:00").

        Returns:
            pandas.Series | float: Time elapsed since the start of the game, in minutes. If data is a DataFrame, return a Series; if data is a Series, return a float.

    """  # fmt: skip

    if isinstance(data, pd.Series):
        qtr = data["qtr"]
        time = data["time"]
    elif isinstance(data, pd.DataFrame):
        qtr = data.loc[:, "qtr"]
        time = data.loc[:, "time"]
    else:
        msg = "Must input data to calc_game_time_elapsed as DataFrame or Series"
        raise TypeError(msg)

    if isinstance(time, float):
        minutes = "15"
        seconds = "0"
    else:
        minutes = time.split(":")[0]
        seconds = time.split(":")[1]
    time_rem_in_qtr = int(minutes) + int(seconds) / 60
    time_elapsed = round((qtr - 1) * 15 + 15 - time_rem_in_qtr, 2)

    return time_elapsed


def clean_team_names(team_names: list[str] | str, year: int) -> list[str]:
    """Modifies list of team names to process data for. Primarily used to generate a list of team names to replace the "all" placeholder.

        If a list of team names is input, no change is made.

        Args:
            team_names (str | list): Either "all" or a list of full team names (e.g. ["Arizona Cardinals", "Baltimore Ravens", ...]).
            year (int): Season to use to generate the full list of team names.

        Returns:
            list: List of full team names.

    """  # fmt: skip

    # Processing only needed if team_names=='all'
    if team_names == "all":
        # Adjust team names and abbreviations for year (team name changes over the years)
        team_abbrs.adjust_team_names(
            [team_abbrs.pbp_abbrevs, team_abbrs.boxscore_website_abbrevs, team_abbrs.roster_website_abbrevs],
            year,
        )
        # Take all team names from dict
        team_names = list(team_abbrs.roster_website_abbrevs.keys())
    # If team_names is a string, convert to list
    elif isinstance(team_names, str):
        team_names = [team_names]

    return team_names


def compute_team_record(scores_df):
    """Computes the record (wins, losses, and ties) of a team GOING INTO each game (i.e. not including the result of the current game).

        Args:
            scores_df (pandas.DataFrame): DataFrame with a row for each game, including the score of each game.
                Must contain all games from Week 1 through the current (or final) week, with no skipped games, for each team being handled (may be one team at a time or the entire league at once).

        Returns:
            pandas.DataFrame: Input DataFrame, with three columns added: "Team Wins", "Team Losses", "Team Ties". These correspond to the team's record GOING INTO the game.

    """  # fmt: skip

    # Add columns tracking whether each game is a win, loss, or tie for the team
    scores_df["Tie"] = scores_df["total_home_score"] == scores_df["total_away_score"]
    scores_df["Win Loc"] = scores_df.apply(lambda x: "Home" if x["total_home_score"] >= x["total_away_score"] else "Away", axis=1)
    scores_df["Win"] = (scores_df["Win Loc"] == scores_df["Site"]) & (np.logical_not(scores_df["Tie"]))
    scores_df["Loss"] = (scores_df["Win Loc"] != scores_df["Site"]) & (np.logical_not(scores_df["Tie"]))
    with pd.option_context("future.no_silent_downcasting", True):
        # 1s and 0s instead of True/False, so we can add them up next
        scores_df = scores_df.replace(to_replace=[True, False], value=[1, 0])

    # Track team record heading into each week
    scores_df = scores_df.sort_values(by="Week")

    # Get team record following each week
    scores_df["Postgame Wins"] = scores_df.groupby("Team Abbrev")["Win"].transform(pd.Series.cumsum)
    scores_df["Postgame Losses"] = scores_df.groupby("Team Abbrev")["Loss"].transform(pd.Series.cumsum)
    scores_df["Postgame Ties"] = scores_df.groupby("Team Abbrev")["Tie"].transform(pd.Series.cumsum)

    # Manipulate to instead show the team's record going into each week
    games_played_by_team = scores_df.groupby("Team Abbrev")["Week"].nunique()
    scores_df = scores_df.set_index(["Team Abbrev", "Week"]).sort_index()
    for new_col, old_col in zip(["Team Wins", "Team Losses", "Team Ties"], ["Postgame Wins", "Postgame Losses", "Postgame Ties"]):
        scores_df[new_col] = __shift_val_one_game_back(scores_df[old_col].to_list(), games_played_by_team)
    scores_df = scores_df.reset_index()

    return scores_df


def construct_game_id(data):
    """Generates Game ID as used by nfl-verse from a set of game information.

        Args:
            data (pandas.Series | dict): Data for a game, including the following fields/keys:
                - Year
                - Week
                - Team (abbreviation)
                - Opponent (opposing team abbreviation)
                - Site ("Home" or "Away")

        Returns:
            str: Game ID for specific game, as used by nfl-verse. Format is "{year}_{week}_{awayteam}_{hometeam}", ex: "2021_01_ARI_TEN"

    """  # fmt: skip

    # Gather/format data from Series
    year = data["Year"]
    week = str(data["Week"]).rjust(2, "0")
    try:
        home_team = data["Home Team"]
        away_team = data["Away Team"]
    except KeyError:
        home_team = data["Team"] if (data["Site"] == "Home" or data["Site"] is True) else data["Opponent"]
        away_team = data["Team"] if (data["Site"] == "Away" or data["Site"] is False) else data["Opponent"]

    # Construct and return game_id string
    game_id = f"{year}_{week}_{away_team}_{home_team}"
    return game_id


def subsample_game_time(player_stats_df, game_times):
    """Reduces the size of the midgame stats data by sampling the data at discrete game times, rather than keeping the stats after every single play.

        Rounds the game time at the start of each play to the nearest value in game_times, then keeps the first instance of each value, as well as the final row (to capture final stats).
        Note that there may be an additional (redundant) row for the final stat line.

        Args:
            player_stats_df (pandas.DataFrame): Player's accumulated stat line after every single play in the game.
            game_times (list | str): List of game times to include in the output. If string is input (such as "all"), no filtering occurs.

        Returns:
            pandas.DataFrame: Player's accumulated stat line at each designated game time. May have an additional (redundant) row for the final stat line.

    """  # fmt: skip

    if not isinstance(game_times, str):
        player_stats_df["Rounded Time"] = player_stats_df.index.map(lambda x: game_times[abs(game_times - float(x)).argmin()])
        # Suppress future warnings about concatenating empty/N/A dataframes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            player_stats_df = pd.concat(
                (
                    player_stats_df.iloc[1:].drop_duplicates(subset="Rounded Time", keep="first"),
                    player_stats_df.iloc[-1].to_frame().T,
                ),
            )

        player_stats_df = (
            player_stats_df.reset_index(drop=True).rename(columns={"Rounded Time": "Elapsed Time"}).set_index("Elapsed Time")
        )

    return player_stats_df


# PRIVATE FUNCTIONS


def __shift_val_one_game_back(postgame_vals_by_team_week, games_played_by_team):
    """Converts a team's wins, losses, or ties over the course of a season from post-game to pre-game values.

        Can process multiple teams simultaneously, or one game at a time (current use case).

        Args:
            postgame_vals_by_team_week (list): List of postgame wins OR losses OR ties for all weeks/teams currently being processed
            games_played_by_team (pandas.Series): Number of games played by each team currently being processed

        Returns:
            list: List of pregame wins OR losses OR ties for all weeks/teams currently being processed

    """  # fmt: skip

    last_game_played_by_team = [x - 1 for x in games_played_by_team.cumsum().to_list()]
    last_game_played_by_team.reverse()
    # Remove last games played by each team
    for ind in last_game_played_by_team:
        del postgame_vals_by_team_week[ind]
    # Add a zero at the beginning of each team's season
    zero_indices = [0, *games_played_by_team.cumsum().iloc[0:-1].to_list()]
    for ind in zero_indices:
        postgame_vals_by_team_week[ind:ind] = [0]

    return postgame_vals_by_team_week
