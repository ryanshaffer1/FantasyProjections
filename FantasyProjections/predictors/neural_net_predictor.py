"""Creates and exports classes to be used as one approach to predicting NFL stats and Fantasy Football scores.

    Classes:
        NeuralNetPredictor : child of FantasyPredictor. Predicts NFL player stats using a Neural Net.
        NeuralNet : PyTorch-based Neural Network object with a custom network architecture.
"""

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
    """Predictor of NFL players' stats in games, using a Neural Net to generate predictions.
    
        Sub-class of FantasyPredictor.

        Args:
            name (str): name of the predictor object, used for logging/display purposes.
            save_folder (str, optional): path to the folder to save model and optimizer settings. Defaults to None.
            load_folder (str, optional): path to the folder to load model and optimizer settings. Defaults to None.
            max_epochs (int, optional): number of training iterations before stopping training. Defaults to 100
            n_epochs_to_stop (int, optional): number of training iterations to check for improvement before stopping. Defaults to 5.
                ex. if a NeuralNetPredictor's performance has not improved over its last n training epochs,
                the training process will be terminated.
        
        Additional Class Attributes:
            device (str): name of the device (e.g. cpu, gpu) used for NeuralNetwork processing
            model (NeuralNetwork): Neural Network implemented via PyTorch
            optimizer (torch.optim.sgd.SGD): Optimizer for model, implemented via PyTorch
                Currently only supports SGD (Stochastic Gradient Descent) type optimizer
        
        Public Methods:
            configure_for_training : Sets up DataLoader, NeuralNetwork, and optimizer objects to train with provided implementation details.
            eval_model : Generates predicted stats for an input evaluation dataset, as computed by the NeuralNetwork.
            load : Initializes a NeuralNetwork and optimizer using specifications saved to file.
            print : Displays the NeuralNetwork architecture and parameter size to console or to a logger.
            save : Stores NeuralNetwork and optimizer specifications to file.
            train_and_validate : Carries out the training process and generates predictions for a separate evaluation dataset.
    """

    # CONSTRUCTOR
    save_folder: str = None
    load_folder: str = None
    max_epochs: int = 100
    n_epochs_to_stop: int = 5

    def __post_init__(self):
        # Evaluates as part of the Constructor.
        # Generates attributes that are not simple data copies of inputs.

        # Assign a device
        self.device = self.__assign_device()
        # Initialize model
        if self.load_folder:
            self.model, self.optimizer = self.load(self.load_folder,print_loaded_model=True)
        else:
            self.model = NeuralNetwork().to(self.device)
            self.optimizer = torch.optim.SGD(self.model.parameters())


    # PUBLIC METHODS

    def configure_for_training(self, param_set, training_data, eval_data, print_model_flag=False):
        """Sets up DataLoader, NeuralNetwork, and optimizer objects to train with provided implementation details.

            Args:
                param_set (HyperParameterSet): set of hyper-parameters used in Neural Network training. Must include:
                    mini_batch_size: size of training data batches used in SGD training iterations
                    learning_rate: Learning Rate used in SGD optimizer
                    lmbda: L2 Regularization Parameter used in SGD optimizer
                training_data (StatsDataset): data to use for Neural Net training
                eval_data (StatsDataset): data to use for Neural Net evaluation (e.g. validation or test data)
                print_model_flag (bool, optional): displays Neural Network model architecture to console or a logger. 
                    Defaults to False.

            Side Effects:
                self.model is given a new instance of NeuralNetwork
                self.optimizer is given a new instance of torch.optim.SGD, using parameters in param_set
                    (specifically, learning_rate and lmbda [weight_decay])

            Returns:
                DataLoader: Training DataLoader to use in training. Batched per mini_batch_size, and shuffled.
                DataLoader: Evaluation DataLoader to use in results evaluation. Not batched or shuffled.
        """
        # Configure data loaders
        train_dataloader = DataLoader(training_data, batch_size=int(param_set.get('mini_batch_size').value), shuffle=True)
        eval_dataloader = DataLoader(eval_data, batch_size=int(eval_data.x_data.shape[0]), shuffle=False)

        # Initialize Neural Network object, configure optimizer, optionally print info on model
        self.model = NeuralNetwork().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=param_set.get('learning_rate').value,
                                         weight_decay=param_set.get('lmbda').value)

        if print_model_flag:
            self.print(self.model, log=True)

        return train_dataloader, eval_dataloader


    def eval_model(self, eval_data=None, eval_dataloader=None):
        """Generates predicted stats for an input evaluation dataset, as computed by the NeuralNetwork.

            Either eval_data or eval_dataloader must be input. 
            If both are input, eval_dataloader will be used.

            Args:
                eval_data (Dataset, optional): data to use for Neural Net evaluation (e.g. validation or test data). Defaults to None.
                eval_dataloader (DataLoader, optional): data to use for Neural Net evaluation (e.g. validation or test data). Defaults to None.

            Returns:
                PredictionResult: Object packaging the predicted and true stats together, which can be used for plotting, 
                    performance assessments, etc.
        """

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
        """Initializes a NeuralNetwork and optimizer using specifications saved to file.

            Assumes the file name for the model is "model.pth"
            And the file name for the optimizer is "opt.pth"

            Args:
                model_folder (str): path where "model.pth" and "optimizer.pth" are located
                print_loaded_model (bool, optional): displays Neural Network model architecture to console or a logger. 
                    Defaults to False.

            Returns:
                NeuralNetwork: Neural Network implemented via PyTorch. 
                    Size/architecture and initial parameters determined by the loaded model.pth
                torch.optim.SGD: Optimizer for model, implemented via PyTorch. 
                    Parameters are determined by the loaded optimizer.pth
        """

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
        """Displays the NeuralNetwork architecture and parameter size to console or to a logger.
            
            Args:
                model (NeuralNetwork): Neural Network implemented via PyTorch.
                log (bool, optional): True if output should be directed to a logger, 
                    False if printed to console. Defaults to False.
        """

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
        """Stores NeuralNetwork and optimizer specifications to file.

            The folder to use is specified by the NeuralNetPredictor's save_folder attribute.
            The NeuralNet model is always saved as "model.pth".
            The optimizer is always saved as "opt.pth"        
        """

        # Check that folder exists, and set filenames
        create_folders(self.save_folder)
        model_save_file = self.save_folder + 'model.pth'
        opt_save_file = self.save_folder + 'opt.pth'
        # Save Neural Net model and optimizer
        torch.save(self.model.state_dict(), model_save_file)
        torch.save(self.optimizer.state_dict(),opt_save_file)
        logger.info(f'Saved PyTorch Model State to {model_save_file}')


    def train_and_validate(self, param_set, train_dataloader, validation_dataloader):
        """Carries out the training process and generates predictions for a separate evaluation dataset.

            The model is first trained using train_dataloader and the input loss function (as well as 
            model and optimizer parameters contained in the NeuralNetPredictor object).
            After each training epoch, the validation data is used to generate prediction accuracy results.
            This process is repeated until either the max epochs condition or the end learning (model no longer
            improving) condition are met.

            Args:
                param_set (HyperParameterSet): set of hyper-parameters used in Neural Network training. Must include:
                    loss_fn: str denoting the loss function to use (e.g. "nn.MSELoss")
                train_dataloader (DataLoader): data to use for Neural Net training.
                validation_dataloader (DataLoader): data to use for Neural Net validation.

            Returns:
                list: accuracy (quantified as average absolute prediction error) of the model's predictions on the
                    validation dataset after each epoch of training.
        """
        # Training/validation loop
        val_perfs = []
        for t in range(self.max_epochs):
            logger.info(f'Training Epoch {t+1}:')
            # Train
            self.__train(train_dataloader, param_set.get('loss_fn').value)

            # Validation
            val_result = self.eval_model(eval_dataloader=validation_dataloader)
            val_perfs.append(np.mean(val_result.diff_pred_vs_truth(absolute=True)))

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
        # Implements Stochastic Gradient Descent to train the model
        # against the provided training dataloader. Periodically logs losses from the loss function
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


class NeuralNetwork(nn.Module):
    """Neural Network model implemented using PyTorch.nn, tailored to the data structures and needs of fantasy_projections.

        Attributes:
            stats_inds (list): lazy, list of indices in input vector corresponding to "raw game stats"
            position_inds (list): lazy, list of indices in input vector corresponding to player positions
            players_inds (list): lazy, list of indices in input vector corresponding to player IDs
            teams_inds (list): lazy, list of indices in input vector corresponding to team IDs
            opponents_inds (list): lazy, list of indices in input vector corresponding to opponent IDs
            embedding_player (nn.Sequential): network layer embedding the player ID information. Parallel to other embedding layers.
            embedding_team (nn.Sequential): network layer embedding the team ID information. Parallel to other embedding layers.
            embedding_opp (nn.Sequential): network layer embedding the opponent ID information. Parallel to other embedding layers.
            linear_stack (nn.Sequential): network layer(s) containing interconnections between embeddings/other stats and output layer.

        Public Methods:
            forward : Defines the feedforward flow of information through the network, including the embedding and linear stack layers.
    """

    # CONSTRUCTOR
    def __init__(self,shape=None):
        """Initializes a NeuralNetwork with a network architecture defined in FantasyProjections project docs.

            Args:
                shape (dict, optional): number of neurons in each layer of the network, keyed by the names of each layer. 
                Defaults to dict default_shape.
                
        """
        super().__init__()

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
        """Defines the feedforward flow of information through the network, including the embedding and linear stack layers.

            Args:
                x (tensor): input vector into Neural Net

            Returns:
                tensor: output vector from Neural Net based on provided input
        """

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
        # Returns the next range of length "length", starting from the end of a previous range "prev_range"
        return range(max(prev_range) + 1, max(prev_range) + 1 + length)
