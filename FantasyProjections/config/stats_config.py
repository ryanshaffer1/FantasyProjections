"""Contains variables defining the configuration of data used in Fantasy Projections.

    Variables:
        default_stat_list (list): list of statistics tracked for all NFL players/games
        default_norm_thresholds (dict): Map for each numeric variable passed into StatsDataset, including (but not limited to) stats in the stat list.
            Also includes variables such as Elapsed Time, Team Wins, etc.
            Each value is a list of the minimum and maximum expected values for the variable. During normalization, these bounds will be mapped to values of 0 and 1.
        default_scoring_weights (dict): Fantasy Points per unit of each statistic, i.e. the Fantasy Football scoring rules used.
        labels_df_to_sleeper (dict): Maps stats names used throughout the project to corresponding stats names used in the Sleeper API.
        labels_df_to_pfr (dict): Maps stats names used throughout the project to corresponding stats names used on pro-football-reference.com (PFR).
        labels_df_to_odds (dict): Maps stats names used throughout the project to corresponding stat names used by The Odds.
"""

default_stat_list = [
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
    "Pass Att"  : 0,
    "Pass Cmp"  : 0,
    "Pass Yds"  : 0.04,
    "Pass TD"   : 4,
    "Int"       : -2,
    "Rush Att"  : 0,
    "Rush Yds"  : 0.1,
    "Rush TD"   : 6,
    "Rec"       : 1,
    "Rec Yds"   : 0.1,
    "Rec TD"    : 6,
    "Fmb"       : -2,
}

labels_df_to_sleeper = {
    "Pass Att": "pass_att",
    "Pass Cmp": "pass_cmp",
    "Pass Yds": "pass_yd",
    "Pass TD": "pass_td",
    "Int": "pass_int",
    "Rush Att": "rush_att",
    "Rush Yds": "rush_yd",
    "Rush TD": "rush_td",
    "Rec": "rec",
    "Rec Yds": "rec_yd",
    "Rec TD": "rec_td",
    "Fmb": "fum_lost",
}

labels_df_to_pfr = {
    "Pass Att": "pass_att",
    "Pass Cmp": "pass_cmp",
    "Pass Yds": "pass_yds",
    "Pass TD": "pass_td",
    "Int": "pass_int",
    "Rush Att": "rush_att",
    "Rush Yds": "rush_yds",
    "Rush TD": "rush_td",
    "Rec": "rec",
    "Rec Yds": "rec_yds",
    "Rec TD": "rec_td",
    "Fmb": "fumbles_lost",
}

labels_df_to_odds = {
    "Pass Att": "player_pass_attempts",
    "Pass Cmp": "player_pass_completions",
    "Pass Yds": "player_pass_yds",
    "Pass TD": "player_pass_tds",
    "Int": "player_pass_interceptions",
    "Rush Att": "player_rush_attempts",
    "Rush Yds": "player_rush_yds",
    "Rush TD": "player_rush_tds",
    "Rec": "player_receptions",
    "Rec Yds": "player_reception_yds",
    "Rec TD": "player_reception_tds",
}
