"""Contains variables defining the configuration of data used in Fantasy Projections.

    Variables: 
        default_stat_list (list): list of statistics tracked for all NFL players/games
        default_norm_thresholds (dict): Map for each numeric variable passed into StatsDataset, including (but not limited to) stats in the stat list.
            Also includes variables such as Elapsed Time, Team Wins, etc.
            Each value is a list of the minimum and maximum expected values for the variable. During normalization, these bounds will be mapped to values of 0 and 1.
        default_scoring_weights (dict): Fantasy Points per unit of each statistic, i.e. the Fantasy Football scoring rules used.
"""

default_stat_list = [
    'Pass Att',
    'Pass Cmp',
    'Pass Yds',
    'Pass TD',
    'Int',
    'Rush Att',
    'Rush Yds',
    'Rush TD',
    'Rec',
    'Rec Yds',
    'Rec TD',
    'Fmb'
                     ]


default_norm_thresholds = {
    "Elapsed Time": [0, 60],
    "Team Score": [0, 100],
    "Opp Score": [0, 100],
    "Possession": [0, 1],
    "Field Position": [0, 100],
    "Pass Att": [0, 100],
    "Pass Cmp": [0, 100],
    "Pass Yds": [-50, 1000],
    "Pass TD": [0, 8],
    "Int": [0, 8],
    "Rush Att": [0, 100],
    "Rush Yds": [-50, 1000],
    "Rush TD": [0, 8],
    "Rec": [0, 100],
    "Rec Yds": [-50, 1000],
    "Rec TD": [0, 8],
    "Fmb": [0, 8],
    "Age": [0, 60],
    "Site": [0, 1],
    "Team Wins": [0, 18],
    "Team Losses": [0, 18],
    "Team Ties": [0, 18],
    "Opp Wins": [0, 18],
    "Opp Losses": [0, 18],
    "Opp Ties": [0, 18],
}


default_scoring_weights = {
    'Pass Att'  : 0,
    'Pass Cmp'  : 0,
    'Pass Yds'  : 0.04,
    'Pass TD'   : 4,
    'Int'       : -2,
    'Rush Att'  : 0,
    'Rush Yds'  : 0.1,
    'Rush TD'   : 6,
    'Rec'       : 1,
    'Rec Yds'   : 0.1,
    'Rec TD'    : 6,
    'Fmb'       : -2,
}
