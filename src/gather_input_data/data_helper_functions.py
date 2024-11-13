import pandas as pd


def stats_to_fantasy_points(stat_line, indices=None):
    # Scoring rules in fantasy format
    fantasy_rules = {
        "pass_ypp": 25,
        "pass_td": 4,
        "int": -2,
        "rush_ypp": 10,
        "rush_td": 6,
        "ppr": 1,
        "rec_ypp": 10,
        "rec_td": 6,
        "fmb": -2,
    }

    if indices:
        stat_line = pd.Series(stat_line, index=indices)

    # Passing
    pass_points = (
        stat_line["Pass Yds"] / fantasy_rules["pass_ypp"]
        + stat_line["Pass TD"] * fantasy_rules["pass_td"]
        + stat_line["Int"] * fantasy_rules["int"]
    )

    # Rushing
    rush_points = (
        stat_line["Rush Yds"] / fantasy_rules["rush_ypp"]
        + stat_line["Rush TD"] * fantasy_rules["rush_td"]
    )

    # Receiving
    rec_points = (
        stat_line["Rec"] * fantasy_rules["ppr"]
        + stat_line["Rec Yds"] / fantasy_rules["rec_ypp"]
        + stat_line["Rec TD"] * fantasy_rules["rec_td"]
    )

    # Misc.
    misc_points = stat_line["Fmb"] * fantasy_rules["fmb"]

    return pass_points + rush_points + rec_points + misc_points


def calc_time_elapsed(data):
    # Calculates game-time elapsed, given a DataFrame with columns for qtr and
    # time
    qtr = data["qtr"]
    time = data["time"]
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


def adjust_team_names(dictionary, year):
    # Note: the team name changes must be included below in reverse
    # chronological order (most important for teams w/ multiple changes, e.g.
    # Washington)

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

    return dictionary
