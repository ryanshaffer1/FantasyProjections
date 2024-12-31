"""Contains variables defining the team names and team abbreviations used in various NFL statistics data sources, as well as functions to manipulate these variables.

    Variables: 
        pbp_abbrevs (dict): Maps full team name to the abbreviation used in play-by-play data from nflverse.
        boxscore_website_abbrevs (dict): Maps full team name to the abbreviation used in boxscore data from pro-football-reference.com.
        roster_website_abbrevs (dict): Maps full team name to the abbreviation used in roster URLs from pro-football-reference.com.

    Functions:
        invert : Swaps keys and values of a dictionary.
        convert_abbrev : Converts a team abbreviation from one format (dictionary) to another format (dictionary).
"""

pbp_abbrevs = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS'
}

boxscore_website_abbrevs = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GNB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KAN',
    'Las Vegas Raiders': 'LVR',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NWE',
    'New Orleans Saints': 'NOR',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SFO',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TAM',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS'
}

roster_website_abbrevs = {
    'Arizona Cardinals': 'crd',
    'Atlanta Falcons': 'atl',
    'Baltimore Ravens': 'rav',
    'Buffalo Bills': 'buf',
    'Carolina Panthers': 'car',
    'Chicago Bears': 'chi',
    'Cincinnati Bengals': 'cin',
    'Cleveland Browns': 'cle',
    'Dallas Cowboys': 'dal',
    'Denver Broncos': 'den',
    'Detroit Lions': 'det',
    'Green Bay Packers': 'gnb',
    'Houston Texans': 'htx',
    'Indianapolis Colts': 'clt',
    'Jacksonville Jaguars': 'jax',
    'Kansas City Chiefs': 'kan',
    'Las Vegas Raiders': 'rai',
    'Los Angeles Chargers': 'sdg',
    'Los Angeles Rams': 'ram',
    'Miami Dolphins': 'mia',
    'Minnesota Vikings': 'min',
    'New England Patriots': 'nwe',
    'New Orleans Saints': 'nor',
    'New York Giants': 'nyg',
    'New York Jets': 'nyj',
    'Philadelphia Eagles': 'phi',
    'Pittsburgh Steelers': 'pit',
    'San Francisco 49ers': 'sfo',
    'Seattle Seahawks': 'sea',
    'Tampa Bay Buccaneers': 'tam',
    'Tennessee Titans': 'oti',
    'Washington Commanders': 'was'
}

def invert(dictionary):
    """Swaps keys and values of a dictionary.

        Args:
            dictionary (dict): Dictionary to invert

        Returns:
            dict: Inverted dictionary: all values within original dict are now keys, and vice versa.
    """
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
    """
    if isinstance(abbrevs, list):
        team_names = [invert(dict1)[abbrev] for abbrev in abbrevs]
        new_abbrevs = [dict2[team_name] for team_name in team_names]
        return new_abbrevs

    team_name = invert(dict1)[abbrevs]
    new_abbrev = dict2[team_name]
    return new_abbrev
