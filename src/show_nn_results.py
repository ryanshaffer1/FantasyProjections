import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import CustomDataset
from nn_model_functions import test, results_eval
from nn_plot_functions import eval_single_games
from neural_network import NeuralNetwork


MODEL_FILE = 'models/model_11072024202956.pth'

# Data files
PBP_DATAFILE = 'data/for_nn/pbp_data_to_nn.csv'
BOXSCORE_DATAFILE = 'data/for_nn/boxscore_data_to_nn.csv'
ID_DATAFILE = 'data/for_nn/data_ids.csv'
# training_data = CustomDataset(pbp_datafile, bs_datafile, id_datafile, years=range(2021,2022))
# training_data.concat(CustomDataset(pbp_datafile, bs_datafile, id_datafile, years=[2023], weeks=range(1,12)))
# train_dataloader = DataLoader(training_data, batch_size=training_data.x_data.shape[0], shuffle=False)
test_data = CustomDataset(
    PBP_DATAFILE,
    BOXSCORE_DATAFILE,
    ID_DATAFILE,
    years=[2023],
    weeks=range(
        15,
        18))
test_dataloader = DataLoader(
    test_data,
    batch_size=test_data.x_data.shape[0],
    shuffle=False)

eval_data = test_data
eval_dataloader = test_dataloader

# Set the model back to the highest performing config
DEVICE = 'cpu'
model = NeuralNetwork().to(DEVICE)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
loss_fn = nn.MSELoss()

# ---------------------
# Final Model Performance Evaluation
# ---------------------
print('\n\n--------- Final Model Performance Test ---------')
# Test
stat_predicts, stat_truths, test_perf = test(eval_dataloader, model)

# Evaluate Test Results
results_eval(stat_predicts, stat_truths, eval_data)

# Show timelines for specific games/players:
N_RANDOM = 10
eval_single_games(stat_predicts, stat_truths, eval_data, n_random=N_RANDOM)

plt.show()
