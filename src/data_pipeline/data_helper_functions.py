import pandas as pd
from data_pipeline import team_abbreviations


def calc_game_time_elapsed(data):
    # Calculates game-time elapsed, given a DataFrame with columns for qtr and
    # time
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


def swap_team_names(year, dictionary, year_threshold, before_name, after_name):
    if year < year_threshold:
        if after_name in dictionary.keys():
            dictionary[before_name] = dictionary[after_name]
            del dictionary[after_name]
    else:
        if before_name in dictionary.keys():
            dictionary[after_name] = dictionary[before_name]
            del dictionary[before_name]

    return dictionary


def adjust_team_names(dictionaries, year):
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
        if "Oakland Raiders" in dictionary.keys() and dictionary["Oakland Raiders"] in [
            "LV",
            "LVR",
        ]:
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

def filter_team_weeks(all_game_info_df, all_rosters_df, full_team_name, year):
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
