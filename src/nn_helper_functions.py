import numpy as np
import pandas as pd
import torch

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
        print(f'Warning: {col.name} not explicitly normalized')

    return col


def unnormalize_stat(col):
    if col.name in thresholds:
        # Unfortunately can't undo the clipping from 0 to 1 performed during
        # normalization, so there's a small chance of lost info...
        [lwr, upr] = thresholds[col.name]
        col = col * (upr - lwr) + lwr
    else:
        print(f'Warning: {col.name} not explicitly normalized')

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


def end_learning(perfs, n_epochs_to_stop):
    # Determines whether to stop training (if test performance has stagnated)
    # Returns true if learning should be stopped
    # If n_epochs_to_stop is less than zero, this feature is turned off
    # (always returns False)
    # Performance must improve by this factor in n_epochs_to_stop in order to
    # continue training
    improvement_threshold = 0.01
    return (n_epochs_to_stop > 0
            and len(perfs) > n_epochs_to_stop
            and perfs[-1] - perfs[-n_epochs_to_stop - 1] >= -improvement_threshold * perfs[-1])


def reformat_sleeper_stats(stat_dict):
    stat_indices_df_to_sleeper = {
        'Pass Att': 'pass_att',
        'Pass Cmp': 'pass_cmp',
        'Pass Yds': 'pass_yd',
        'Pass TD': 'pass_td',
        'Int': 'pass_int',
        'Rush Att': 'rush_att',
        'Rush Yds': 'rush_yd',
        'Rush TD': 'rush_td',
        'Rec': 'rec',
        'Rec Yds': 'rec_yd',
        'Rec TD': 'rec_td',
        'Fmb': 'fum_lost'
    }
    stat_line = []
    for sleeper_stat_label in stat_indices_df_to_sleeper.values():
        stat_value = stat_dict.get(sleeper_stat_label, 0)
        stat_line.append(stat_value)

    return stat_line


def assign_device(print_device=True):
    # Get cpu, gpu or mps device for training.
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
    if print_device:
        print(f'Using {device} device')

    return device


def print_model(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    print('')
    print(model)
    print(f'Total tunable parameters: {total_params}')


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
