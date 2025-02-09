# Make source code importable from sandbox folder
import sys
import os
sys.path.append(os.getcwd()+'/FantasyProjections')

import numpy as np
import pandas as pd
import _pickle
from config import nn_config
from misc.dataset import StatsDataset
from predictors import NeuralNetPredictor


models_folder = 'models/'
# All folders in model folder
folder_contents = os.listdir(models_folder)
folders = []
for item in folder_contents:
    if os.path.isdir(os.path.join(models_folder,item)):
        folders.append(models_folder+item+'/')

# Read in datasets
PBP_DATAFILE = 'data/to_nn/midgame_data_to_nn.csv'
BOXSCORE_DATAFILE = 'data/to_nn/final_stats_to_nn.csv'
ID_DATAFILE = 'data/to_nn/data_ids.csv'
pbp_df = pd.read_csv(PBP_DATAFILE, engine='pyarrow')
boxscore_df = pd.read_csv(BOXSCORE_DATAFILE, engine='pyarrow')
id_df = pd.read_csv(ID_DATAFILE, engine='pyarrow')
all_data = StatsDataset('All', id_df=id_df, pbp_df=pbp_df, boxscore_df=boxscore_df)
test_data = all_data.slice_by_criteria(inplace=False, years=[2023], weeks=range(15,18))
test_data.name = 'Test'
test_data_pregame = test_data.slice_by_criteria(inplace=False,elapsed_time=[0])
test_data_pregame.name = 'Test (Pre-Game)'

# Test NN within each folder in the models folder
perfs_dict = {}
for i, folder in enumerate(folders):
    print(f'{i+1} of {len(folders)}')
    try:
        neural_net = NeuralNetPredictor(name='Neural Net', load_folder=folder, **nn_config.nn_train_settings)
    except FileNotFoundError:
        print(f'no model found in {folder}.')
    except _pickle.UnpicklingError:
        print(f'incompatible model found in {folder}.')
    nn_result_all_test = neural_net.eval_model(eval_data=test_data)
    nn_result_pregame = neural_net.eval_model(eval_data=test_data_pregame)
    perfs_dict[folder] = [np.mean(nn_result_all_test.diff_pred_vs_truth(absolute=True)),
                          np.mean(nn_result_pregame.diff_pred_vs_truth(absolute=True))]

perfs_df = pd.DataFrame(perfs_dict).transpose().rename(columns={0:'all data',1:'pregame'})
perfs_df.to_csv('all models comparison.csv')