import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset import CustomDataset
from nn_helper_functions import stats_to_fantasy_points
from nn_model_functions import results_eval

# Prediction Algorithm: The expected fantasy score of a player in a given
# game is equal to their fantasy score from their previous game

# Data sources
PBP_DATAFILE = 'data/for_nn/pbp_data_to_nn.csv'
BOXSCORE_DATAFILE = 'data/for_nn/boxscore_data_to_nn.csv'
ID_DATAFILE = 'data/for_nn/data_ids.csv'

# Training, validation, and test datasets
# training_data = CustomDataset(pbp_datafile,boxscore_datafile,id_datafile,weeks=range(1,15))
# validation_data = CustomDataset(pbp_datafile,boxscore_datafile,id_datafile,weeks=range(15,17))
test_data = CustomDataset(
    PBP_DATAFILE,
    BOXSCORE_DATAFILE,
    ID_DATAFILE,
    weeks=range(
        17,
        19))
all_data = CustomDataset(PBP_DATAFILE, BOXSCORE_DATAFILE, ID_DATAFILE)

# Organize data
eval_data = test_data

# Drop all the duplicated rows that are for the same game, and only
# dependent on elapsed game time - that variable is irrelevant here, so we
# can greatly simplify
duplicated_rows_eval_data = eval_data.id_data.reset_index().duplicated(
    subset=['Player', 'Year', 'Week'])
eval_data.y_data = eval_data.y_data[np.logical_not(duplicated_rows_eval_data)]
eval_data.id_data = eval_data.id_data.reset_index(
    drop=True).loc[np.logical_not(duplicated_rows_eval_data)].reset_index(drop=True)
duplicated_rows_all_data = all_data.id_data.duplicated(
    subset=['Player', 'Year', 'Week'])
all_data.y_data = all_data.y_data[np.logical_not(duplicated_rows_all_data)]
all_data.id_data = all_data.id_data.loc[np.logical_not(duplicated_rows_all_data)].reset_index(
    drop=True)

# Variables needed to search for previous games and convert stats to
# fantasy points
first_year_in_dataset = min(all_data.id_data['Year'])
stat_indices = [
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

# Set up dataframe to use to search for previous games
all_IDs = all_data.id_data.copy()[['Player', 'Year', 'Week']]
all_IDs = all_IDs.reset_index().set_index(
    ['Player', 'Year', 'Week']).sort_index()
indices = all_IDs.index
all_IDs['Player Copy'] = all_IDs.index.get_level_values(0).to_list()
all_IDs['Prev Year'] = all_IDs.index.get_level_values(1).to_list()
all_IDs['Prev Week'] = all_IDs.index.get_level_values(2).to_list()
all_IDs['Prev Game Found'] = False
all_IDs['Continue Prev Search'] = True
while any(all_IDs['Continue Prev Search']):
    # Only make changes to rows of the dataframe that are continuing the search
    search_IDs = all_IDs['Continue Prev Search']
    # Find the next (previous) week to look for
    all_IDs.loc[search_IDs,
                'Prev Week'] = all_IDs.loc[search_IDs,
                                           'Prev Week'] - 1
    # If previous week is set to Week 0, fix this by going to Week 18 of the
    # previous year
    all_IDs.loc[search_IDs, 'Prev Year'] = all_IDs.loc[search_IDs].apply(
        lambda x: x['Prev Year'] - 1 if x['Prev Week'] == 0 else x['Prev Year'], axis=1)
    all_IDs.loc[search_IDs, 'Prev Week'] = all_IDs.loc[search_IDs].apply(
        lambda x: 18 if x['Prev Week'] == 0 else x['Prev Week'], axis=1)

    # Check if previous game is in the dataframe
    all_IDs.loc[search_IDs, 'Prev Game Found'] = all_IDs.loc[search_IDs].apply(
        lambda x: (x['Player Copy'], x['Prev Year'], x['Prev Week']) in indices, axis=1)

    # Check whether to continue searching for a previous game for this player
    all_IDs.loc[search_IDs,'Continue Prev Search'] = np.logical_not(
        all_IDs.loc[search_IDs,'Prev Game Found']) & (
            all_IDs.loc[search_IDs,'Prev Year'] >= first_year_in_dataset)

# Assign the index (row) of the previous game
all_IDs['Prev Game Index'] = np.nan
valid_rows = all_IDs['Prev Game Found']
all_IDs.loc[valid_rows, 'Prev Game Index'] = all_IDs.loc[valid_rows].apply(
    lambda x: all_IDs.loc[(x['Player Copy'], x['Prev Year'], x['Prev Week']), 'index'], axis=1)

# Get the stats for each previous game, in order
all_IDs['tensor'] = all_IDs.apply(
    lambda x: list(all_data.y_data[int(x['Prev Game Index'])]) if x['Prev Game Index'] >= 0
    else [np.nan] * all_data.y_data.shape[1], axis=1)

# Grab stats for each game in the evaluation data
prev_game_stats_df = eval_data.id_data.apply(
    lambda x: all_IDs.loc[(x['Player'], x['Year'], x['Week']), 'tensor'], axis=1)
prev_game_stats = torch.tensor(prev_game_stats_df.to_list())

# Un-normalize and compute Fantasy score
stat_predicts = stats_to_fantasy_points(
    prev_game_stats, stat_indices, normalized=True)
stat_truths = stats_to_fantasy_points(
    eval_data.y_data,
    stat_indices,
    normalized=True)

# Evaluate results
# Calculate average difference in fantasy points between prediction and truth
fp_predicts = stat_predicts['Fantasy Points']
fp_truths = stat_truths['Fantasy Points']
fp_diff = [abs(predict - truth)
           for predict, truth in zip(fp_predicts, fp_truths)]
# Display average fantasy point difference
print(
    f"Test Error: Avg Fantasy Points Different = {(np.nanmean(fp_diff)):>0.2f}")
# Generate more detailed evaluation of results
results_eval(stat_predicts, stat_truths, eval_data)


plt.show()
