from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from nn_helper_functions import end_learning, save_hp_tuning_results
from nn_model_functions import train, test, results_eval
from nn_plot_functions import plot_grid_search_results
from hyper_tuner import HyperParameter, HyperParameterSet
from dataset import CustomDataset
from neural_network import NeuralNetwork

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

program_start_time = datetime.now()

# ---------------------
# Parameters
# ---------------------


# User Preferences (don't affect the actual process)
DISP_RUN_TIMES = False

# Parameters affecting the outcome of the run, not automatically tuned
OPTIMIZE_HYPERS = True
HYPER_TUNER_LAYERS = 2
HYPER_TUNER_STEPS_PER_DIM = 5
MAX_EPOCHS = 100
N_EPOCHS_TO_STOP = 5
SCALE_EPOCHS_OVER_LAYERS = True # If True, max_epochs and n_epochs_to_stop will double with each layer of the hyperparameter grid search

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
                            #    val_range=[5,50],
                               val_range=[1e-2,100],
                               val_scale='log',
                               num_steps=HYPER_TUNER_STEPS_PER_DIM)
lmbda = HyperParameter('lmbda',
                       optimizable=True,
                       init_value=0,
                    #    val_range=[1e-7,1e-5],
                       val_range=[1e-7,1e-3],
                       val_scale='log',
                       num_steps=HYPER_TUNER_STEPS_PER_DIM)
loss_fn = HyperParameter('loss_fn',
                         optimizable=False,
                         init_value=nn.MSELoss()
                         # if optimizable, add kwargs: ,val_range=[nn.MSELoss(),nn.CrossEntropyLoss()],val_scale='selection')
                         )

# Set of all hyper-parameters (collected so that they can be varied/optimized together)
param_set = HyperParameterSet((mini_batch_size,learning_rate,lmbda,loss_fn),optimize=OPTIMIZE_HYPERS)

# ---------------------
# Data Setup
# ---------------------
# Data files
PBP_DATAFILE = 'data/for_nn/pbp_data_to_nn.csv'
BOXSCORE_DATAFILE = 'data/for_nn/boxscore_data_to_nn.csv'
ID_DATAFILE = 'data/for_nn/data_ids.csv'
pbp_df = pd.read_csv(PBP_DATAFILE)
boxscore_df = pd.read_csv(BOXSCORE_DATAFILE)
id_df = pd.read_csv(ID_DATAFILE)


# Training, validation, and test datasets
# training_data = CustomDataset(pbp_datafile,bs_datafile,id_datafile,num_to_use=182955)
training_data = CustomDataset(pbp_df, boxscore_df, id_df, years=range(2021,2022))
training_data.concat(CustomDataset(pbp_df, boxscore_df, id_df, years=[2023], weeks=range(1,12)))
validation_data = CustomDataset(pbp_df, boxscore_df, id_df, years=[2023], weeks=range(12,15))
test_data = CustomDataset(pbp_df, boxscore_df, id_df, years=[2023], weeks=range(15,18))

for (dataset_name,dataset) in zip(('Training Data','Validataion Data','Test Data'),(training_data,validation_data,test_data)):
    print(f'{dataset_name} size: {dataset.x_data.shape[0]}')

# Output file
model_file = f'models/model_{datetime.strftime(datetime.now(),'%m%d%Y%H%M%S')}.pth'
hp_gridpoints_file = f'models/hyper_grid_{datetime.strftime(datetime.now(),'%m%d%Y%H%M%S')}.csv'

filename = 'models/hyper_grid_11142024164752.csv'
plot_grid_search_results(filename,param_set,variables=('learning_rate','lmbda'))
plt.show()

# ---------------------
# One-Time Setup
# ---------------------
# Get cpu, gpu or mps device for training.
DEVICE = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)
print(f'Using {DEVICE} device')

# Adjust variables if not optimizing hyper-parameters
if not OPTIMIZE_HYPERS:
    HYPER_TUNER_LAYERS = 1

# ---------------------
# Iterate through HyperParameter tuning loop
# ---------------------
hyper_tuning_table = [] # Array to keep track of all hyperparameter gridpoints run
for tune_layer in range(HYPER_TUNER_LAYERS):
    print(f'\nOptimization Round {tune_layer+1} of {HYPER_TUNER_LAYERS}\n-------------------------------')
    # Iterate through all combinations of hyperparameters
    gridpoint_model_perf = [] # List of model performance values for all combos of hyperparameter values
    for grid_ind in range(param_set.total_gridpoints):
        # Set and display hyperparameters for current run
        param_set.set_values(grid_ind)
        print(f'\nHP Grid Point {grid_ind+1} of {param_set.total_gridpoints}: -------------------- ')
        for hp in param_set.hyper_parameters:
            print(f"\t{hp.name} = {hp.value}")

        # Configure data loaders
        train_dataloader = DataLoader(training_data, batch_size=int(mini_batch_size.value), shuffle=False)
        validation_dataloader = DataLoader(validation_data, batch_size=int(mini_batch_size.value), shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=int(mini_batch_size.value), shuffle=False)

        # Initialize Neural Network object
        model = NeuralNetwork().to(DEVICE)
        # Set up nn model parameters and optimizer
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate.value,weight_decay=lmbda.value)
        if not OPTIMIZE_HYPERS:
            print('')
            print(model)
            print(f'Total tunable parameters: {total_params}')

        # ---------------------
        # Model Training and Validation Testing
        # ---------------------
        # Training/validation loop
        val_perfs = []
        for t in range(MAX_EPOCHS):
            print(f'Training Epoch {t+1} ------------- ')
            # Train
            trainStartTime = datetime.now()
            train(train_dataloader, model, DEVICE, loss_fn.value, optimizer)
            if DISP_RUN_TIMES:
                print(f'Training Time: {datetime.now()-trainStartTime}')

            # Validation
            valStartTime = datetime.now()
            stat_predicts,stat_truths,val_perf = test(validation_dataloader, model)
            val_perfs.append(val_perf)
            stop_training = end_learning(val_perfs,N_EPOCHS_TO_STOP) # Check stopping condition
            if DISP_RUN_TIMES:
                print(f'Val. Test Time: {datetime.now()-valStartTime}')

            # Check stopping condition
            if stop_training:
                print('Learning has stopped, terminating training process')
                break

        # Track validation performance for the set of hyperparameters used
        gridpoint_model_perf.append(val_perf)

        # Save the model if it is the best performing so far
        if grid_ind == np.nanargmin(gridpoint_model_perf):
            torch.save(model.state_dict(), model_file)


    min_grid_index = np.nanargmin(gridpoint_model_perf)
    if OPTIMIZE_HYPERS:
        hyper_tuning_table = save_hp_tuning_results(param_set,gridpoint_model_perf,hyper_tuning_table,tune_layer,hp_gridpoints_file)
        print(
            f'Layer {tune_layer+1} '
            f'Complete. Optimal performance: '
            f'{gridpoint_model_perf[min_grid_index]}. '
            f'Hyper-parameters used: '
            )
        for hp in param_set.hyper_parameters:
            print(f"\t{hp.name} = {hp.values[min_grid_index]}")
    if tune_layer < HYPER_TUNER_LAYERS-1:
        param_set.refine_grid(min_grid_index)
        if SCALE_EPOCHS_OVER_LAYERS:
            MAX_EPOCHS*= 2
            N_EPOCHS_TO_STOP*= 2
        print('Beginning next hyper-parameter optimization iteration.')
    else:
        print('Model Training Complete!')

# Set the model back to the highest performing config
model.load_state_dict(torch.load(model_file,weights_only=True))

# ---------------------
# Final Model Performance Evaluation
# ---------------------
print('\n\n--------- Final Model Performance Test ---------')
# Test
testStartTime = datetime.now()
stat_predicts,stat_truths,test_perf = test(test_dataloader, model)
if DISP_RUN_TIMES:
    print(f'Test Time: {datetime.now()-testStartTime}')

# Evaluate Test Results
evalStartTime = datetime.now()
results_eval(stat_predicts,stat_truths,test_data)
if len(hyper_tuning_table) > 0:
    plot_grid_search_results(hp_gridpoints_file,param_set)
if DISP_RUN_TIMES:
    print(f'Eval Time: {datetime.now()-evalStartTime}')

torch.save(model.state_dict(), model_file)
print(f'Saved PyTorch Model State to {model_file}')

print(f'Done! Elapsed Time: {datetime.now()-program_start_time}')


plt.show()
print('done')
