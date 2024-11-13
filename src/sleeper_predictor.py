import json
from sleeper_wrapper import Stats, Players
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import CustomDataset
from nn_helper_functions import reformat_sleeper_stats, stats_to_fantasy_points
from nn_model_functions import results_eval

# Prediction Algorithm: Pulls projections from Sleeper Fantasy. Not sure how they compute their projections!
# Sleeper API calls documented here:
# https://github.com/SwapnikKatkoori/sleeper-api-wrapper

# User flags
REFRESH_PLAYERS = False

# Data files (Sleeper)
SLEEPER_PLAYER_DICT_FILE = 'data/misc/sleeper_player_dict.json'
SLEEPER_PROJ_DICT_FILE = 'data/misc/sleeper_projections_dict.json'

# Data sources (truth)
PBP_DATAFILE = 'data/for_nn/pbp_data_to_nn.csv'
BOXSCORE_DATAFILE = 'data/for_nn/boxscore_data_to_nn.csv'
ID_DATAFILE = 'data/for_nn/data_ids.csv'

# Training, validation, and test datasets
# all_data = CustomDataset(pbp_datafile, boxscore_datafile, id_datafile)
# training_data = CustomDataset(pbp_datafile,boxscore_datafile,id_datafile,weeks=range(1,15))
# validation_data = CustomDataset(pbp_datafile,boxscore_datafile,id_datafile,weeks=range(15,17))
test_data = CustomDataset(
    PBP_DATAFILE, BOXSCORE_DATAFILE, ID_DATAFILE,
    years=[2023], weeks=range(15,18))

# Dataset to evaluate against
eval_data = test_data

# PROCESS TRUTH DATA
# Drop all the duplicated rows that are for the same game, and only
# dependent on elapsed game time - that variable is irrelevant here, so we
# can greatly simplify
duplicated_rows_eval_data = eval_data.id_data.reset_index().duplicated(subset=[
    'Player', 'Year', 'Week'])
eval_data.y_data = eval_data.y_data[np.logical_not(duplicated_rows_eval_data)]
eval_data.id_data = eval_data.id_data.reset_index(
    drop=True).loc[np.logical_not(duplicated_rows_eval_data)].reset_index(drop=True)

# Un-normalize and compute Fantasy score
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
stat_truths = stats_to_fantasy_points(
    eval_data.y_data,
    stat_indices,
    normalized=True)

# Unique year-week combinations in evaluation dataset
eval_data.id_data['Year-Week'] = eval_data.id_data[['Year',
                                                    'Week']].astype(str).agg('-'.join, axis=1)
unique_year_weeks = list(eval_data.id_data['Year-Week'].unique())

# PROCESS DATA FROM SLEEPER
# Get dictionary of all players in the league
if REFRESH_PLAYERS:
    players = Players()
    player_dict = players.get_all_players()

    # Re-organize player dict into dictionary mapping full names to Sleeper player IDs
    # THIS DOESN'T WORK --- MULTIPLE PLAYERS WITH SAME NAME AND DIFFERENT IDS.
    # EX: MIKE WILLIAMS
    player_to_sleeperID = {}
    for player in player_dict:
        sleeper_id = player
        player_name = player_dict[player].get('full_name', None)
        if player_name:
            player_to_sleeperID[player_name] = sleeper_id
        else:
            print(f'Warning: {player} not added to player dictionary')

    # Save player dictionary to JSON file for use next time
    with open(SLEEPER_PLAYER_DICT_FILE, 'w', encoding='utf-8') as file:
        json.dump(player_to_sleeperID, file)
else:
    # Load the JSON data into a Python dictionary
    with open(SLEEPER_PLAYER_DICT_FILE, 'r', encoding='utf-8') as file:
        player_to_sleeperID = json.load(file)

# Gather all stats from Sleeper
with open(SLEEPER_PROJ_DICT_FILE, 'r', encoding='utf-8') as file:
    all_proj_dict = json.load(file)
if not all([year_week in all_proj_dict for year_week in unique_year_weeks]):
    # Gather any unsaved stats from Sleeper
    stats = Stats()
    for year_week in unique_year_weeks:
        if year_week not in all_proj_dict:
            [year, week] = year_week.split('-')
            week_proj = stats.get_week_projections(
                'regular', int(year), int(week))
            all_proj_dict[year_week] = week_proj
            print(
                f'Adding Year-Week {year_week} to Sleeper projections dictionary: {SLEEPER_PROJ_DICT_FILE}')
    # Save player dictionary to JSON file for use next time
    with open(SLEEPER_PROJ_DICT_FILE, 'w', encoding='utf-8') as file:
        json.dump(all_proj_dict, file)

# Build up array of predicted stats for all eval_data cases based on
# sleeper projections dictionary
stat_predicts = torch.empty(0)
for row in range(eval_data.id_data.shape[0]):
    id_row = eval_data.id_data.iloc[row]
    year_week = id_row['Year-Week']
    player = id_row['Player']
    if player in player_to_sleeperID:
        proj_stats = all_proj_dict[year_week][player_to_sleeperID[player]]
        stat_line = torch.tensor(reformat_sleeper_stats(proj_stats))
    else:
        stat_line = torch.zeros([12])
    stat_predicts = torch.cat((stat_predicts, stat_line))


# Compute fantasy points using stat lines (note that this ignores the
# built-in fantasy points projection in the Sleeper API, which differs
# from the sum of the stats)
stat_predicts = stats_to_fantasy_points(torch.reshape(
    stat_predicts, [-1, 12]), stat_indices, normalized=False)

# EVALUATE RESULTS
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
