"""Creates multiple Fantasy Football stat predictors (using various prediction algorithms), evaluates their predictions for a user-configurable
    set of past NFL games, and logs/visualizes the results in multiple user-configurable plots.
    Contains many user settings for the Neural Network Predictor, which is the most custom (and most important) type of predictor in the project.
    
    This module is a script to be run alone. Before running, the packages in requirements.txt must be installed. Use the following terminal command:
    > pip install -u requirements.txt
"""

from datetime import datetime
import logging
import logging.config
import pandas as pd
import matplotlib.pyplot as plt

from config.log_config import LOGGING_CONFIG
from config import data_vis_config
from config import hp_config
from config import nn_config

from misc.dataset import StatsDataset
from misc.manage_files import move_logfile

from neural_net import HyperParameterSet

from tuners import GridSearchTuner
from predictors import NeuralNetPredictor, SleeperPredictor, PerfectPredictor, LastNPredictor
from results import PredictionResultGroup, PredictionResult
from gamblers import BasicGambler

# Output files
FOLDER_PREFIX = ''
save_folder = f'models/{FOLDER_PREFIX}{datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S')}/'
LOAD_FOLDER = 'models/11222024003003/'

# Neural Net Data files
PBP_DATAFILE = 'data/to_nn/midgame_data_to_nn.csv'
BOXSCORE_DATAFILE = 'data/to_nn/final_stats_to_nn.csv'
ID_DATAFILE = 'data/to_nn/data_ids.csv'

# Sleeper Data files
SLEEPER_PROJ_DICT_FILE = 'data/misc/sleeper_projections_dict.json'

# Set up logger
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')

# Start
logger.info('Starting Program')

# Read data files
pbp_df = pd.read_csv(PBP_DATAFILE, engine='pyarrow')
boxscore_df = pd.read_csv(BOXSCORE_DATAFILE, engine='pyarrow')
id_df = pd.read_csv(ID_DATAFILE, engine='pyarrow')
logger.info('Input files read')
for name,file in zip(['pbp','boxscore','IDs'],[PBP_DATAFILE, BOXSCORE_DATAFILE, ID_DATAFILE]):
    logger.debug(f'{name}: {file}')

# Training, validation, and test datasets
# split_datasets(training={2021:'all',2022:'all',2023:range(1,12)},
#                validation={2023:range(12,15)},
#                test={2023:range(15,18)})
all_data = StatsDataset('All', id_df=id_df, pbp_df=pbp_df, boxscore_df=boxscore_df)
training_data = all_data.slice_by_criteria(inplace=False, years=range(2021,2022))
training_data.concat(all_data.slice_by_criteria(inplace=False, years=[2023], weeks=range(1,12)))
training_data.name = 'Training'
validation_data = all_data.slice_by_criteria(inplace=False, years=[2023], weeks=range(12,15))
validation_data.name = 'Validation'
test_data = all_data.slice_by_criteria(inplace=False, years=[2023], weeks=range(15,18))
test_data.name = 'Test'
test_data_pregame = test_data.slice_by_criteria(inplace=False,elapsed_time=[0])
test_data_pregame.name = 'Test (Pre-Game)'
pregame_2024_data = all_data.slice_by_criteria(inplace=False, years=[2024], elapsed_time=[0])
pregame_2024_data.name = '2024 (Pre-Game)'
for dataset in (training_data,validation_data,test_data):
    logger.info(f'{dataset.name} Dataset size: {dataset.x_data.shape[0]}')

# Hyper-Parameters affecting the behavior of the Neural Net
param_set = HyperParameterSet(hp_dict=hp_config.hp_defaults)

# Initialize and train neural net
# neural_net = NeuralNetPredictor(name='Neural Net', load_folder=LOAD_FOLDER, **nn_config.nn_train_settings)
neural_net = NeuralNetPredictor(name='Neural Net', save_folder=save_folder, **nn_config.nn_train_settings)

if hp_config.hp_tuner_settings['optimize_hypers']:
    # Tuning algorithm for Neural Net Hyper-Parameters
    param_tuner = GridSearchTuner(param_set, save_file=save_folder+'hyper_tuner.csv', **hp_config.hp_tuner_settings)
    param_tuner.tune_hyper_parameters(eval_function=neural_net.train_and_validate,
                                    save_function=neural_net.save, reset_function=neural_net.load,
                                    eval_kwargs = {'training_data':training_data, 'validation_data':validation_data},
                                    reset_kwargs = {'model_folder':save_folder},
                                    plot_variables = ['learning_rate','lmbda'])
else:
    neural_net.train_and_validate(training_data=training_data, validation_data=validation_data, param_set=param_set)

# Alternate predictors
sleeper_predictor = SleeperPredictor(name='Sleeper',
                                     proj_dict_file=SLEEPER_PROJ_DICT_FILE,
                                     update_players=False) # Create Sleeper prediction model
naive_predictor = LastNPredictor(name='Last N Games Predictor', n=3) # Create Naive prediction model
perfect_predictor = PerfectPredictor(name='Perfect Predictor') # Create Perfect prediction model

# Evaluate Model(s) against Test Data
nn_result = neural_net.eval_model(eval_data=test_data)
sleeper_result = sleeper_predictor.eval_model(eval_data=test_data_pregame)
naive_result = naive_predictor.eval_model(eval_data=test_data_pregame, all_data=all_data)
perfect_result = perfect_predictor.eval_model(eval_data=test_data)

# Gamble based on prediction results, and evaluate success
nn_result = neural_net.eval_model(eval_data=pregame_2024_data)
naive_result = naive_predictor.eval_model(eval_data=pregame_2024_data, all_data=all_data)
gambler = BasicGambler(nn_result)
gambler.plot_earnings()
logger.info(f'Earnings from gambling: {gambler.earnings: 0.2f} units ({gambler.accuracy*100 :0.2f}% accurate)')


# Plot evaluation results
all_results = PredictionResultGroup((nn_result,))
all_results.plot_all(PredictionResult.plot_error_dist, together=True, absolute=True)
all_results.plot_all(PredictionResult.plot_single_games, n_random=5)
all_results.plot_all(PredictionResult.plot_scatters, data_vis_config.scatter_plot_settings)

# Move logfile to the correct folder
logging.shutdown()
move_logfile('logfile.log', save_folder)

# Display plots
plt.show()
print('Program Complete')
