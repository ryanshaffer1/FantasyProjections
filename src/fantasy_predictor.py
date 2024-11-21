from datetime import datetime
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from dataset import CustomDataset
from hyper_tuner import HyperParameter, HyperParameterSet, GridSearchTuner
from predictors import NeuralNetPredictor, SleeperPredictor, NaivePredictor, PerfectPredictor
from prediction_result import PredictionResultGroup, PredictionResult

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


# User Preferences (don't affect the actual process)
DISP_RUN_TIMES = False

# Output files
model_file = f'models/model_{datetime.strftime(datetime.now(),'%m%d%Y%H%M%S')}.pth'
hp_gridpoints_file = f'models/hyper_grid_{datetime.strftime(datetime.now(),'%m%d%Y%H%M%S')}.csv'
GOOD_FILE = 'models/model_11142024164752.pth'
BAD_FILE = 'models/model_11182024223717.pth'

# ---------------------
# Data Setup
# ---------------------
# Neural Net Data files
PBP_DATAFILE = 'data/for_nn/pbp_data_to_nn.csv'
BOXSCORE_DATAFILE = 'data/for_nn/boxscore_data_to_nn.csv'
ID_DATAFILE = 'data/for_nn/data_ids.csv'
pbp_df = pd.read_csv(PBP_DATAFILE)
boxscore_df = pd.read_csv(BOXSCORE_DATAFILE)
id_df = pd.read_csv(ID_DATAFILE)

# Sleeper Data files
SLEEPER_PLAYER_DICT_FILE = 'data/misc/sleeper_player_dict.json'
SLEEPER_PROJ_DICT_FILE = 'data/misc/sleeper_projections_dict.json'

# Training, validation, and test datasets
all_data = CustomDataset(pbp_df, boxscore_df, id_df)
training_data = CustomDataset(pbp_df, boxscore_df, id_df, years=range(2021,2022))
training_data.concat(CustomDataset(pbp_df, boxscore_df, id_df, years=[2023], weeks=range(1,12)))
validation_data = CustomDataset(pbp_df, boxscore_df, id_df, years=[2023], weeks=range(12,15))
test_data = CustomDataset(pbp_df, boxscore_df, id_df, years=[2023], weeks=range(15,18))
test_data_pregame = test_data.slice_by_criteria(inplace=False,elapsed_time=[0])

for (dataset_name,dataset) in zip(('Training Data','Validataion Data','Test Data'),(training_data,validation_data,test_data)):
    print(f'{dataset_name} size: {dataset.x_data.shape[0]}')


# Parameters affecting the outcome of the run, not automatically tuned
hp_tuner_settings = {
    'optimize_hypers': False,
    'hyper_tuner_layers': 2,
    'hyper_tuner_steps_per_dim': 5,
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
                                init_value=1000,
                                #  val_range = [1000,10000],
                                #  val_range=[100,10000],
                                #  val_scale='log',
                                #  num_steps=HYPER_TUNER_STEPS_PER_DIM
                                )
learning_rate = HyperParameter('learning_rate',
                            optimizable=True,
                            init_value=50,
                            # val_range=[5,50],
                            val_range=[1e-2,100],
                            val_scale='log',
                            num_steps=hp_tuner_settings['hyper_tuner_steps_per_dim'])
lmbda = HyperParameter('lmbda',
                    optimizable=True,
                    init_value=0,
                    # val_range=[1e-7,1e-5],
                    val_range=[1e-7,1e-3],
                    val_scale='log',
                    num_steps=hp_tuner_settings['hyper_tuner_steps_per_dim'])
loss_fn = HyperParameter('loss_fn',
                        optimizable=False,
                        init_value=nn.MSELoss()
                        # if optimizable, add kwargs: ,val_range=[nn.MSELoss(),nn.CrossEntropyLoss()],val_scale='selection')
                        )

# Set of all hyper-parameters (collected so that they can be varied/optimized together)
param_set = HyperParameterSet((mini_batch_size,learning_rate,lmbda,loss_fn),optimize=hp_tuner_settings['optimize_hypers'])
param_tuner = GridSearchTuner(param_set,hp_gridpoints_file,hp_tuner_settings)

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
# neural_net1 = NeuralNetPredictor('Bad Neural Net', nn_settings, load_file=BAD_FILE)
# neural_net2 = NeuralNetPredictor('Improved Neural Net', nn_settings, load_file=GOOD_FILE)
neural_net = NeuralNetPredictor('Neural Net', nn_settings, save_file=model_file)
param_tuner.tune_neural_net(neural_net, training_data, validation_data)

# Create Sleeper prediction model
sleeper_predictor = SleeperPredictor('Sleeper',
                                     SLEEPER_PLAYER_DICT_FILE,
                                     SLEEPER_PROJ_DICT_FILE,
                                     update_players=False)

# Create Naive prediction model
naive_predictor = NaivePredictor('Naive: Previous Game')

# Create Perfect prediction model
perfect_predictor = PerfectPredictor('Perfect Predictor')

# Evaluate Model(s) against Test Data
nn_result = neural_net.eval_model(eval_data=test_data)
# sleeper_result = sleeper_predictor.eval_model(eval_data=test_data_pregame)
# naive_result = naive_predictor.eval_model(eval_data=test_data_pregame, all_data=all_data)
# perfect_result = perfect_predictor.eval_model(eval_data=test_data)

# Plot evaluation results
# all_results = PredictionResultGroup((nn_result, sleeper_result, naive_result, perfect_result))
all_results = PredictionResultGroup((nn_result,))
all_results.plot_all(PredictionResult.plot_error_dist, together=True, absolute=True)
# all_results.plot_all(PredictionResult.plot_single_games, n_random=0)
all_results.plot_all(PredictionResult.plot_scatters, scatter_plot_settings)


plt.show()
print('done')
