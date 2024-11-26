from datetime import datetime
import logging
import logging.config
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt

from config.log_config import LOGGING_CONFIG
from misc.dataset import CustomDataset
from misc.manage_files import move_logfile
from misc.prediction_result import PredictionResultGroup, PredictionResult

from tuners.hyper_tuner import HyperParameter, HyperParameterSet
from tuners.grid_search_tuner import GridSearchTuner

from predictors.neural_net_predictor import NeuralNetPredictor
from predictors.sleeper_predictor import SleeperPredictor
from predictors.alternate_predictors import LastNPredictor, PerfectPredictor

# To Do:
# - add as a variable in x: depth chart position (ie WR1 vs WR4 on roster) - makes it easier for model. Especially for people coming on and off the bench
#   - need to track this weekly - maybe ESPN has that data?
# - endLearning: end learning if last performance score is nan!
# - Kyren Williams, 2023 week 15 - in minute 47, rushing yards is blank (leads to big drop in points). Something wrong with pbp parser?
# - Need to track across data sources by a common player ID, not name. Example: Marquise Brown == Hollywood Brown. Rosters from input dataset include sleeper_id, so this will match up with
#       data from Sleeper, thank god. Just need to include this as a new column in ID dataframes, and use that lookup in sleeper_predictor

# Huge issues:
# - training/test data must be chosen more thoughtfully - they are not all random and independent. Same player,
# same game leads to same output, and sometimes the training and test data are probably only seconds apart in game time.
# - no ability to generalize beyond player "word bank" - if a new player shows up that it hasn't been trained on, it will
# have no clue. Consider adding player stats/quantitative data so that player comps could in theory be made
# - should input/output data be normalized (as it currently is)? Is this producing more error/training difficulty?

# Think these are fixed, but leaving here for now just in case:
# - no "knowledge" of the player - either need to figure out how to pass it player_ID information (so that it can
# "remember" how good various players are) or a ton of additional quantitative data to characterize the player.
# height, weight, draft position, historical stats over career, etc.
# - no "knowledge" of the team besides team record going into the game - same solution space as above. How to
# correlate the performance of teammates? I.e. good quarterback -> better stats for WR?

# ---------------------
# Parameters
# ---------------------


# Output files
FOLDER_PREFIX = ''
save_folder = f'models/{FOLDER_PREFIX}{datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S')}/'
LOAD_FOLDER = 'models/11222024003003/'
GOOD_FILE = 'models/model_11142024164752.pth'
BAD_FILE = 'models/model_11182024223717.pth'

# ---------------------
# Data Setup
# ---------------------
# Neural Net Data files
PBP_DATAFILE = 'data2/to_nn/midgame_data_to_nn.csv'
BOXSCORE_DATAFILE = 'data2/to_nn/final_stats_to_nn.csv'
ID_DATAFILE = 'data2/to_nn/data_ids.csv'

# Sleeper Data files
SLEEPER_PLAYER_DICT_FILE = 'data/misc/sleeper_player_dict.json'
SLEEPER_PROJ_DICT_FILE = 'data/misc/sleeper_projections_dict.json'

# Set up logger
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')

# Start
logger.info('Starting Program')

# Read data files
pbp_df = pd.read_csv(PBP_DATAFILE)
boxscore_df = pd.read_csv(BOXSCORE_DATAFILE)
id_df = pd.read_csv(ID_DATAFILE)
logger.info('Input files read')
for name,file in zip(['pbp','boxscore','IDs'],[PBP_DATAFILE, BOXSCORE_DATAFILE, ID_DATAFILE]):
    logger.debug(f'{name}: {file}')

# Training, validation, and test datasets
all_data = CustomDataset('All', pbp_df, boxscore_df, id_df)
training_data = CustomDataset('Training', pbp_df, boxscore_df, id_df, years=range(2021,2022))
training_data.concat(CustomDataset('', pbp_df, boxscore_df, id_df, years=[2023], weeks=range(1,12)))
validation_data = CustomDataset('Validation', pbp_df, boxscore_df, id_df, years=[2023], weeks=range(12,15))
test_data = CustomDataset('Test', pbp_df, boxscore_df, id_df, years=[2023], weeks=range(15,18))
test_data_pregame = test_data.slice_by_criteria(inplace=False,elapsed_time=[0])
test_data_pregame.name = 'Test (Pre-Game)'

for dataset in (training_data,validation_data,test_data):
    logger.info(f'{dataset.name} Dataset size: {dataset.x_data.shape[0]}')


# Parameters affecting the outcome of the run, not automatically tuned
hp_tuner_settings = {
    'optimize_hypers': False,
    'hyper_tuner_layers': 2,
    'hyper_tuner_steps_per_dim': 2,
    'scale_epochs_over_layers': True, # If True, max_epochs and n_epochs_to_stop will double with each layer of the hyperparameter grid search
    'plot_tuning_results': True,
}
nn_settings = {
    'max_epochs': 100,
    'n_epochs_to_stop': 5,
}

# Hyper-parameters (automatically tuned if optimize_hypers = True)
mini_batch_size = HyperParameter('mini_batch_size',
                                optimizable=False,
                                value=1000,
                                #  val_range = [1000,10000],
                                #  val_range=[100,10000],
                                #  val_scale='log',
                                #  num_steps=HYPER_TUNER_STEPS_PER_DIM
                                )
learning_rate = HyperParameter('learning_rate',
                            optimizable=True,
                            value=50,
                            # val_range=[5,50],
                            val_range=[1e-2,100],
                            val_scale='log',
                            num_steps=hp_tuner_settings['hyper_tuner_steps_per_dim'])
lmbda = HyperParameter('lmbda',
                    optimizable=True,
                    value=0,
                    # val_range=[1e-7,1e-5],
                    val_range=[1e-7,1e-3],
                    val_scale='log',
                    num_steps=hp_tuner_settings['hyper_tuner_steps_per_dim'])
loss_fn = HyperParameter('loss_fn',
                        optimizable=False,
                        value=nn.MSELoss()
                        # if optimizable, add kwargs: ,val_range=[nn.MSELoss(),nn.CrossEntropyLoss()],val_scale='selection')
                        )

# Set of all hyper-parameters (collected so that they can be varied/optimized together)
param_set = HyperParameterSet((mini_batch_size,learning_rate,lmbda,loss_fn),optimize=hp_tuner_settings['optimize_hypers'])
param_tuner = GridSearchTuner(param_set,save_folder,**hp_tuner_settings)

# Configure scatter plots
scatter_plot_settings = []
scatter_plot_settings.append({'columns': ['Pass Att',
                                'Pass Cmp',
                                'Pass Yds',
                                'Pass TD',
                                'Int'],
                    'slice': {'Position': ['QB']},
                    'subtitle': 'Passing Stats',
                    'histograms': True})
scatter_plot_settings.append({'columns': ['Pass Att', 'Pass Cmp', 'Pass Yds', 'Pass TD'],
                    'legend_slice': {'Position': [['QB'], ['RB', 'WR', 'TE']]},
                    'subtitle': 'Passing Stats',
                    })
scatter_plot_settings.append({'columns': ['Rush Att', 'Rush Yds', 'Rush TD', 'Fmb'],
                    'subtitle': 'Rushing Stats',
                    'histogram': True
                    })
scatter_plot_settings.append({'columns': ['Rec', 'Rec Yds', 'Rec TD'],
                    'legend_slice': {'Position': [['RB', 'WR', 'TE'], ['QB']]},
                    'subtitle': 'Receiving Stats',
                    })
scatter_plot_settings.append({'columns': ['Fantasy Points'],
                    'subtitle': 'Fantasy Points',
                    'histograms': True
                    })

# Initialize and train neural net
# neural_net1 = NeuralNetPredictor(name='Bad Neural Net', load_file=BAD_FILE, **nn_settings)
neural_net = NeuralNetPredictor(name='Neural Net', load_folder=LOAD_FOLDER, save_folder=save_folder, **nn_settings)
# neural_net = NeuralNetPredictor(name='Neural Net', save_folder=save_folder, **nn_settings)
# param_tuner.tune_neural_net(neural_net, training_data, validation_data)

# Create Sleeper prediction model
sleeper_predictor = SleeperPredictor(name='Sleeper',
                                     player_dict_file=SLEEPER_PLAYER_DICT_FILE,
                                     proj_dict_file=SLEEPER_PROJ_DICT_FILE,
                                     update_players=False)

# Create Naive prediction model
naive_predictor = LastNPredictor(name='Naive: Previous Game', n=3)

# Create Perfect prediction model
perfect_predictor = PerfectPredictor(name='Perfect Predictor')

# Evaluate Model(s) against Test Data
nn_result = neural_net.eval_model(eval_data=test_data)
sleeper_result = sleeper_predictor.eval_model(eval_data=test_data_pregame)
naive_result = naive_predictor.eval_model(eval_data=test_data_pregame, all_data=all_data)
perfect_result = perfect_predictor.eval_model(eval_data=test_data)

# Plot evaluation results
all_results = PredictionResultGroup((nn_result,))
all_results.plot_all(PredictionResult.plot_error_dist, together=True, absolute=True)
all_results.plot_all(PredictionResult.plot_single_games, n_random=0)
all_results.plot_all(PredictionResult.plot_scatters, scatter_plot_settings)

# Move logfile to the correct folder
logger.info(f'Saving logfile to {save_folder}')
logging.shutdown()
move_logfile('logfile.log',save_folder)

# Display plots
plt.show()
print('Program Complete')
