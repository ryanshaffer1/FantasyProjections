"""Contains variables defining the team names and team abbreviations used in various NFL statistics data sources, as well as functions to manipulate these variables.

    Variables:
        pbp_abbrevs (dict): Maps full team name to the abbreviation used in play-by-play data from nflverse.
        boxscore_website_abbrevs (dict): Maps full team name to the abbreviation used in boxscore data from pro-football-reference.com.
        roster_website_abbrevs (dict): Maps full team name to the abbreviation used in roster URLs from pro-football-reference.com.

    Functions:
        invert : Swaps keys and values of a dictionary.
        convert_abbrev : Converts a team abbreviation from one format (dictionary) to another format (dictionary).
        adjust_team_names : Changes NFL team names and abbreviations in internal dictionaries to account for the changing of NFL team names over the years in real life.
        swap_team_names : Changes the NFL team name used as a key in the dictionary if the name is not appropriate for the year.

"""  # fmt: skip

pbp_abbrevs = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


boxscore_website_abbrevs = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GNB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KAN",
    "Las Vegas Raiders": "LVR",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NWE",
    "New Orleans Saints": "NOR",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SFO",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TAM",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


roster_website_abbrevs = {
    "Arizona Cardinals": "crd",
    "Atlanta Falcons": "atl",
    "Baltimore Ravens": "rav",
    "Buffalo Bills": "buf",
    "Carolina Panthers": "car",
    "Chicago Bears": "chi",
    "Cincinnati Bengals": "cin",
    "Cleveland Browns": "cle",
    "Dallas Cowboys": "dal",
    "Denver Broncos": "den",
    "Detroit Lions": "det",
    "Green Bay Packers": "gnb",
    "Houston Texans": "htx",
    "Indianapolis Colts": "clt",
    "Jacksonville Jaguars": "jax",
    "Kansas City Chiefs": "kan",
    "Las Vegas Raiders": "rai",
    "Los Angeles Chargers": "sdg",
    "Los Angeles Rams": "ram",
    "Miami Dolphins": "mia",
    "Minnesota Vikings": "min",
    "New England Patriots": "nwe",
    "New Orleans Saints": "nor",
    "New York Giants": "nyg",
    "New York Jets": "nyj",
    "Philadelphia Eagles": "phi",
    "Pittsburgh Steelers": "pit",
    "San Francisco 49ers": "sfo",
    "Seattle Seahawks": "sea",
    "Tampa Bay Buccaneers": "tam",
    "Tennessee Titans": "oti",
    "Washington Commanders": "was",
}


def invert(dictionary):
    """Swaps keys and values of a dictionary.

        Args:
            dictionary (dict): Dictionary to invert

        Returns:
            dict: Inverted dictionary: all values within original dict are now keys, and vice versa.

    """  # fmt: skip
    return {v: k for k, v in dictionary.items()}


def convert_abbrev(abbrevs, dict1, dict2):
    """Converts a team abbreviation from one format (dictionary) to another format (dictionary).

        Example:
            > convert_abbrev("ARI", pbp_abbrevs, roster_website_abbrevs)
            "crd"

        Args:
            abbrevs (str | list of strs): Team abbreviation(s) as found in dict1
            dict1 (dict): Original format
            dict2 (dict): New format

        Returns:
            str | list of strs: Abbreviation(s) for the same team(s), formatted according to dict2

    """  # fmt: skip
    if isinstance(abbrevs, list):
        team_names = [invert(dict1)[abbrev] for abbrev in abbrevs]
        new_abbrevs = [dict2[team_name] for team_name in team_names]
        return new_abbrevs

    team_name = invert(dict1)[abbrevs]
    new_abbrev = dict2[team_name]
    return new_abbrev


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

    """  # fmt: skip

    # Note: the team name changes must be included below in reverse
    # chronological order (most important for teams w/ multiple changes, e.g.
    # Washington)

    # Handle edge case of singular dict being input
    if isinstance(dictionaries, dict):
        dictionaries = [dictionaries]

    for dictionary in dictionaries:
        # 2022: Washington Football Team -> Washington Commanders
        dictionary = swap_team_names(
            year,
            dictionary,
            2021.5,
            "Washington Football Team",
            "Washington Commanders",
        )

        # 2020: Washington Redskins -> Washington Football Team
        dictionary = swap_team_names(
            year,
            dictionary,
            2019.5,
            "Washington Redskins",
            "Washington Football Team",
        )

        # 2020: Oakland Raiders -> Las Vegas Raiders
        dictionary = swap_team_names(
            year,
            dictionary,
            2019.5,
            "Oakland Raiders",
            "Las Vegas Raiders",
        )
        # also need to make a value (abbreviation) swap for this one
        if "Oakland Raiders" in dictionary and dictionary["Oakland Raiders"] in ["LV", "LVR"]:
            dictionary["Oakland Raiders"] = "OAK"
        if "Las Vegas Raiders" in dictionary and dictionary["Las Vegas Raiders"] == "OAK":
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

    """  # fmt: skip

    if year < year_threshold:
        if after_name in dictionary:
            dictionary[before_name] = dictionary[after_name]
            del dictionary[after_name]
    elif before_name in dictionary:
        dictionary[after_name] = dictionary[before_name]
        del dictionary[before_name]

    return dictionary
