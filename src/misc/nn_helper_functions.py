import logging
import numpy as np
import pandas as pd

# Set up logger
logger = logging.getLogger('log')

thresholds = {
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

# Names of all statistics
stat_indices_default = [
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
    'Fmb']

def normalize_stat(col):
    if col.name in thresholds:
        [lwr, upr] = thresholds[col.name]
        col = (col - lwr) / (upr - lwr)
        col = col.clip(0, 1)
    else:
        logger.warning(f'Warning: {col.name} not explicitly normalized')

    return col


def unnormalize_stat(col):
    if col.name in thresholds:
        # Unfortunately can't undo the clipping from 0 to 1 performed during
        # normalization, so there's a small chance of lost info...
        [lwr, upr] = thresholds[col.name]
        col = col * (upr - lwr) + lwr
    else:
        logger.warning(f'Warning: {col.name} not explicitly normalized')

    return col


def stats_to_fantasy_points(stat_line, stat_indices=None, normalized=False):
    if stat_indices == 'default':
        stat_indices = stat_indices_default

    # Scoring rules in fantasy format
    fantasy_rules = {'pass_ypp': 25,
                     'pass_td': 4,
                     'int': -2,
                     'rush_ypp': 10,
                     'rush_td': 6,
                     'ppr': 1,
                     'rec_ypp': 10,
                     'rec_td': 6,
                     'fmb': -2}

    if stat_indices:
        stat_line = pd.DataFrame(stat_line)
        if len(stat_line.columns) == 1:
            stat_line = stat_line.transpose()
        stat_line.columns = stat_indices
    if normalized:
        for col in stat_line.columns:
            stat_line[col] = unnormalize_stat(stat_line[col])

    # Passing
    pass_points = stat_line['Pass Yds'] / fantasy_rules['pass_ypp'] + stat_line['Pass TD'] * \
        fantasy_rules['pass_td'] + stat_line['Int'] * fantasy_rules['int']

    # Rushing
    rush_points = stat_line['Rush Yds'] / fantasy_rules['rush_ypp'] + \
        stat_line['Rush TD'] * fantasy_rules['rush_td']

    # Receiving
    rec_points = stat_line['Rec'] * fantasy_rules['ppr'] + stat_line['Rec Yds'] / \
        fantasy_rules['rec_ypp'] + stat_line['Rec TD'] * fantasy_rules['rec_td']

    # Misc.
    misc_points = stat_line['Fmb'] * fantasy_rules['fmb']

    # Add fantasy points to stat line
    stat_line['Fantasy Points'] = pass_points + \
        rush_points + rec_points + misc_points

    return stat_line


def remove_game_duplicates(eval_data):
    duplicated_rows_eval_data = eval_data.id_data.reset_index().duplicated(subset=[
        'Player', 'Year', 'Week'])
    eval_data.y_data = eval_data.y_data[np.logical_not(duplicated_rows_eval_data)]
    eval_data.id_data = eval_data.id_data.reset_index(
        drop=True).loc[np.logical_not(duplicated_rows_eval_data)].reset_index(drop=True)
    return eval_data


def gen_random_games(id_df, n_random, game_ids=None):
    if not game_ids:
        game_ids = []
    # (Optionally) Add random players/games to plot
    for _ in range(n_random):
        valid_new_entry = False
        while not valid_new_entry:
            ind = np.random.randint(id_df.shape[0])
            game_dict = {col: id_df.iloc[ind][col]
                        for col in ['Player', 'Week', 'Year']}
            if game_dict not in game_ids:
                game_ids.append(game_dict)
                valid_new_entry = True

    return game_ids
