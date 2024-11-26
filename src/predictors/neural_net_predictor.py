from dataclasses import dataclass
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from misc.nn_helper_functions import stats_to_fantasy_points
from misc.manage_files import create_folders
from .fantasypredictor import FantasyPredictor

# Set up logger
logger = logging.getLogger('log')

@dataclass
class NeuralNetPredictor(FantasyPredictor):
    # CONSTRUCTOR
    save_folder: str = None
    load_folder: str = None
    max_epochs: int = 100
    n_epochs_to_stop: int = 5

    def __post_init__(self):
        self.device = self.__assign_device()
        # Initialize model
        if self.load_folder:
            self.model, self.optimizer = self.load(self.load_folder,print_loaded_model=True)
        else:
            self.model = NeuralNetwork().to(self.device)
            self.optimizer = torch.optim.SGD(self.model.parameters())


    # PUBLIC METHODS

    def configure_for_training(self, param_set, training_data, validation_data, print_model_flag=False):
        # Configure data loaders
        train_dataloader = DataLoader(training_data, batch_size=int(param_set.get('mini_batch_size').value), shuffle=False)
        validation_dataloader = DataLoader(validation_data, batch_size=int(validation_data.x_data.shape[0]), shuffle=False)

        # Initialize Neural Network object, configure optimizer, optionally print info on model
        self.model = NeuralNetwork().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=param_set.get('learning_rate').value,
                                         weight_decay=param_set.get('lmbda').value)

        if print_model_flag:
            self.print(self.model, log=True)

        return train_dataloader, validation_dataloader


    def eval_model(self, eval_data=None, eval_dataloader=None):
        # Create dataloader if only a dataset is passed as input
        if not eval_dataloader:
            eval_dataloader = DataLoader(eval_data, batch_size=int(eval_data.x_data.shape[0]), shuffle=False)

        # Gather all predicted/true outputs for the input dataset
        self.model.eval()
        pred = torch.empty([0, 12])
        y_matrix = torch.empty([0, 12])
        with torch.no_grad():
            for (x_matrix, y_vec) in eval_dataloader:
                pred = torch.cat((pred, self.model(x_matrix)))
                y_matrix = torch.cat((y_matrix, y_vec))

        # Convert outputs into un-normalized statistics/fantasy points
        stat_predicts = stats_to_fantasy_points(pred, stat_indices='default', normalized=True)
        stat_truths = stats_to_fantasy_points(y_matrix, stat_indices='default', normalized=True)
        result = self._gen_prediction_result(stat_predicts, stat_truths, eval_dataloader.dataset)

        return result


    def load(self, model_folder, print_loaded_model=False):
        model_file = model_folder + 'model.pth'
        opt_file = model_folder + 'opt.pth'
        # Establish shape of the model based on data within file
        state_dict = torch.load(model_file, weights_only=True)
        shape = {
        'players_input': state_dict['embedding_player.0.weight'].shape[1],
        'teams_input': state_dict['embedding_team.0.weight'].shape[1],
        'positions_input': 4,
        'embedding_player': state_dict['embedding_player.0.weight'].shape[0],
        'embedding_team': state_dict['embedding_team.0.weight'].shape[0],
        'embedding_opp': state_dict['embedding_opp.0.weight'].shape[0],
        'linear_stack': state_dict['linear_stack.0.weight'].shape[0],
        'stats_output': state_dict['linear_stack.2.weight'].shape[0],
        }
        shape['stats_input'] = (state_dict['linear_stack.0.weight'].shape[1] -
                            shape['embedding_player'] -
                            shape['embedding_team'] -
                            shape['embedding_opp'] -
                            shape['positions_input']
                            )

        # Create Neural Network model with shape determined above
        model = NeuralNetwork(shape=shape).to(self.device)

        # Load weights/biases into model
        model.load_state_dict(state_dict)

        # Initialize optimizer, then load state dict from file
        optimizer = torch.optim.SGD(model.parameters())
        opt_state_dict = torch.load(opt_file, weights_only=True)
        optimizer.load_state_dict(opt_state_dict)

        # Print model
        if print_loaded_model:
            logger.info(f'Loaded model from file {model_file}')
            self.print(model, log=True)

        return model, optimizer


    def print(self, model, log=False):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        if log:
            logger.info(model)
            logger.info(f'Total tunable parameters: {total_params}')
        else:
            print('')
            print(model)
            print(f'Total tunable parameters: {total_params}')

    def save(self):
        # Check that folder exists, and set filenames
        create_folders(self.save_folder)
        model_save_file = self.save_folder + 'model.pth'
        opt_save_file = self.save_folder + 'opt.pth'
        # Save Neural Net model and optimizer
        torch.save(self.model.state_dict(), model_save_file)
        torch.save(self.optimizer.state_dict(),opt_save_file)
        logger.info(f'Saved PyTorch Model State to {model_save_file}')


    def train_and_validate(self, param_set, train_dataloader, validation_dataloader):
        # Training/validation loop
        val_perfs = []
        for t in range(self.max_epochs):
            logger.info(f'Training Epoch {t+1}:')
            # Train
            self.__train(train_dataloader, param_set.get('loss_fn').value)

            # Validation
            val_result = self.eval_model(eval_dataloader=validation_dataloader)
            val_perfs.append(np.mean(val_result.avg_diff(absolute=True)))

            # Check stopping condition
            if self.__end_learning(val_perfs,self.n_epochs_to_stop): # Check stopping condition
                logger.info('Learning has stopped, terminating training process')
                break

        return val_perfs


    # PRIVATE METHODS
    def __assign_device(self, print_device=True):
        # Get cpu, gpu or mps device for training.
        device = (
            'cuda'
            if torch.cuda.is_available()
            else 'mps'
            if torch.backends.mps.is_available()
            else 'cpu'
        )
        if print_device:
            logger.info(f'Using {device} device')

        return device


    def __end_learning(self,perfs, n_epochs_to_stop, improvement_threshold=0.01):
        # Determines whether to stop training (if test performance has stagnated)
        # Returns true if learning should be stopped
        # If n_epochs_to_stop is less than zero, this feature is turned off
        # (always returns False)
        # Performance must improve by this factor in n_epochs_to_stop in order to
        # continue training

        return (len(perfs) > n_epochs_to_stop > 0
                and perfs[-1] - perfs[-n_epochs_to_stop - 1] >= -improvement_threshold * perfs[-1])


    def __train(self, dataloader, loss_fn, print_losses=True):
        size = len(dataloader.dataset)
        num_batches = int(np.ceil(size / dataloader.batch_size))
        self.model.train()
        for batch, (x_matrix, y_matrix) in enumerate(dataloader):
            x_matrix, y_matrix = x_matrix.to(self.device), y_matrix.to(self.device)

            # Compute prediction error
            pred = self.model(x_matrix)
            loss = loss_fn(pred, y_matrix)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if print_losses and batch % int(num_batches / 10) == 0:
                loss, current = loss.item(), (batch + 1) * len(x_matrix)
                logger.debug(f'\tloss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


# Shape of neural network (can be reconfigured during object initialization)
default_shape = {
    'players_input': 300,
    'teams_input': 32,
    'stats_input': 25,
    'positions_input': 4,
    'embedding_player': 50,
    'embedding_team': 10,
    'embedding_opp': 10,
    'linear_stack': 300,
    'stats_output': 12,
}

class NeuralNetwork(nn.Module):
    # CONSTRUCTOR

    def __init__(self,shape=None):
        super().__init__()

        # Establish shape based on optional inputs
        if not shape:
            shape = default_shape
        else:
            for (key,val) in default_shape.items():
                if key not in shape:
                    shape[key] = val
        # Indices of input vector that correspond to each "category" (needed bc they are embedded separately)
        self.stats_inds = range(0, shape['stats_input'])
        self.position_inds = self.__index_range(self.stats_inds, shape['positions_input'])
        self.players_inds = self.__index_range(self.position_inds, shape['players_input'])
        self.teams_inds = self.__index_range(self.players_inds, shape['teams_input'])
        self.opponents_inds = self.__index_range(self.teams_inds, shape['teams_input'])

        self.embedding_player = nn.Sequential(
            nn.Linear(shape['players_input'], shape['embedding_player'], dtype=float),
            nn.ReLU()
        )
        self.embedding_team = nn.Sequential(
            nn.Linear(shape['teams_input'], shape['embedding_team'], dtype=float),
            nn.ReLU()
        )
        self.embedding_opp = nn.Sequential(
            nn.Linear(shape['teams_input'], shape['embedding_opp'], dtype=float),
            nn.ReLU()
        )
        n_input_to_linear_stack = sum(shape[item] for item in ['stats_input','positions_input','embedding_player','embedding_team','embedding_opp'])
        self.linear_stack = nn.Sequential(
            nn.Linear(n_input_to_linear_stack, shape['linear_stack'], dtype=float),
            nn.ReLU(),
            nn.Linear(shape['linear_stack'], shape['stats_output'], dtype=float),
            nn.Sigmoid()
        )

    # PUBLIC METHODS

    def forward(self, x):
        player_embedding = self.embedding_player(x[:, self.players_inds])
        team_embedding = self.embedding_team(x[:, self.teams_inds])
        opp_embedding = self.embedding_opp(x[:, self.opponents_inds])
        x_embedded = torch.cat((x[:,
                                  0:max(self.position_inds) + 1],
                                player_embedding,
                                team_embedding,
                                opp_embedding),
                               dim=1)
        logits = self.linear_stack(x_embedded)
        return logits


    # PRIVATE METHODS

    def __index_range(self, prev_range, length):
        return range(max(prev_range) + 1, max(prev_range) + 1 + length)
