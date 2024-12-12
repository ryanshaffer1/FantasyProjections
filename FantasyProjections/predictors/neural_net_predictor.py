"""Creates and exports classes to be used as one approach to predicting NFL stats and Fantasy Football scores.

    Classes:
        NeuralNetPredictor : child of FantasyPredictor. Predicts NFL player stats using a Neural Net.
"""

from dataclasses import dataclass
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from config.hp_config import default_hp_values
from misc.stat_utils import stats_to_fantasy_points
from misc.manage_files import create_folders
from neural_net import NeuralNetwork
from neural_net import HyperParameterSet
from predictors import FantasyPredictor

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
            nn_shape (dict, optional): Neural Network layers and number of neurons per layer. Defaults to the default Neural Network shape.
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
    nn_shape: dict = None
    max_epochs: int = 100
    n_epochs_to_stop: int = 5

    def __post_init__(self):
        # Evaluates as part of the Constructor.
        # Generates attributes that are not simple data copies of inputs.

        # Assign a device
        self.device = self.__assign_device()
        # Initialize model
        if self.load_folder:
            self.load(self.load_folder,print_loaded_model=True)
        else:
            self.model = NeuralNetwork(shape=self.nn_shape).to(self.device)
            self.optimizer = torch.optim.SGD(self.model.parameters())


    # PUBLIC METHODS

    def configure_for_training(self, training_data, eval_data, param_set=None, **kwargs):
        """Sets up DataLoader, NeuralNetwork, and optimizer objects to train with provided implementation details.

            Args:
                training_data (StatsDataset): data to use for Neural Net training
                eval_data (StatsDataset): data to use for Neural Net evaluation (e.g. validation or test data)
                param_set (HyperParameterSet | dict, optional): set of hyper-parameters used in Neural Network training. 
                    If HyperParameterSet, may include:
                        mini_batch_size (HyperParameter): size of training data batches used in SGD training iterations.
                        learning_rate (HyperParameter): Learning Rate used in SGD optimizer.
                        lmbda (HyperParameter): L2 Regularization Parameter used in SGD optimizer.
                    If dict, may include: 
                        mini_batch_size (int): size of training data batches used in SGD training iterations.
                        learning_rate (float): Learning Rate used in SGD optimizer.
                        lmbda (float): L2 Regularization Parameter used in SGD optimizer.
                    Default values for each optional input:
                        mini_batch_size: 1000
                        learning_rate: 50
                        lmbda: 0

            Keyword-Args:
                reset_model (bool, optional): whether to continue using an existing NN model or re-generate the NN. Defaults to True.
                    Note that the optimizer will be reset either way. If an existing optimizer is desired, set the same hyper-parameters in param_set.
                print_model_flag (bool, optional): displays Neural Network model architecture to console or a logger. 
                    Defaults to False.

            Side Effects (Modified Attributes):
                self.model is given a new instance of NeuralNetwork
                self.optimizer is given a new instance of torch.optim.SGD, using parameters in param_set
                    (specifically, learning_rate and lmbda [weight_decay])

            Returns:
                DataLoader: Training DataLoader to use in training. Batched per mini_batch_size, and shuffled.
                DataLoader: Evaluation DataLoader to use in results evaluation. Not batched or shuffled.
        """

        # Optional keyword arguments
        reset_model = kwargs.get('reset_model', True)
        print_model_flag = kwargs.get('print_model_flag', False)

        # Extract or set values of hyper-parameters
        if isinstance(param_set, HyperParameterSet):
            param_set = param_set.to_dict()
        if param_set is None:
            param_set = {}
        mini_batch_size = int(param_set.get('mini_batch_size', default_hp_values['mini_batch_size']))
        learning_rate = param_set.get('learning_rate', default_hp_values['learning_rate'])
        lmbda = param_set.get('lmbda', default_hp_values['lmbda'])

        # Configure data loaders
        train_dataloader = DataLoader(training_data, batch_size=int(mini_batch_size), shuffle=True)
        eval_dataloader = DataLoader(eval_data, batch_size=int(eval_data.x_data.shape[0]), shuffle=False)

        # Initialize Neural Network object, configure optimizer, optionally print info on model
        if reset_model:
            self.model = NeuralNetwork(shape=self.nn_shape).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=learning_rate,
                                         weight_decay=lmbda)

        if print_model_flag:
            self.print(self.model, log=True)

        return train_dataloader, eval_dataloader


    def eval_model(self, eval_data=None, eval_dataloader=None, **kwargs):
        """Generates predicted stats for an input evaluation dataset, as computed by the NeuralNetwork.

            Either eval_data or eval_dataloader must be input. 
            If both are input, eval_dataloader will be used.

            Args:
                eval_data (StatsDataset, optional): data to use for Neural Net evaluation (e.g. validation or test data). Defaults to None.
                eval_dataloader (DataLoader, optional): data to use for Neural Net evaluation (e.g. validation or test data). Defaults to None.

            Keyword-Args:
                All keyword arguments are passed to the function stats_to_fantasy_points and to the PredictionResult constructor. 
                See the related documentation for descriptions and valid inputs.
                All keyword arguments are optional.

            Returns:
                PredictionResult: Object packaging the predicted and true stats together, which can be used for plotting, 
                    performance assessments, etc.
        """

        # Raise error if normalized=False in kwargs
        if 'normalized' in kwargs and not kwargs['normalized']:
            raise ValueError('NeuralNetPredictor evaluation data must be normalized.')
        kwargs['normalized'] = True

        # Create dataloader if only a dataset is passed as input
        if not eval_dataloader:
            eval_dataloader = DataLoader(eval_data, batch_size=int(eval_data.x_data.shape[0]), shuffle=False)
        # Override data in eval_data (in case both are passed, or eval_data is used in following code)
        eval_data = eval_dataloader.dataset

        # List of stats being used to compute fantasy score
        stat_columns = eval_dataloader.dataset.y_data_columns

        # Gather all predicted/true outputs for the input dataset
        self.model.eval()
        pred = torch.empty([0, len(stat_columns)])
        y_matrix = torch.empty([0, len(stat_columns)])
        with torch.no_grad():
            for (x_matrix, y_vec) in eval_dataloader:
                pred = torch.cat((pred, self.model(x_matrix)))
                y_matrix = torch.cat((y_matrix, y_vec))

        # Convert outputs into un-normalized statistics/fantasy points
        stat_predicts = stats_to_fantasy_points(pred, stat_indices=stat_columns, **kwargs)
        stat_truths = self.eval_truth(eval_dataloader.dataset, **kwargs)
        result = self._gen_prediction_result(stat_predicts, stat_truths, eval_dataloader.dataset, **kwargs)

        return result


    def load(self, model_folder, print_loaded_model=False):
        """Initializes a NeuralNetwork and optimizer using specifications saved to file.

            Assumes the file name for the model is "model.pth"
            And the file name for the optimizer is "opt.pth"

            Args:
                model_folder (str): path where "model.pth" and "optimizer.pth" are located
                print_loaded_model (bool, optional): displays Neural Network model architecture to console or a logger. 
                    Defaults to False.

            Attributes modified:
                model (NeuralNetwork): Neural Network implemented via PyTorch. 
                    Size/architecture and initial parameters determined by the loaded model.pth
                optimizer (torch.optim.SGD): Optimizer for model, implemented via PyTorch. 
                    Parameters are determined by the loaded optimizer.pth
                nn_shape (dict): Neural Network layers and number of neurons per layer.
        """

        model_file = model_folder + 'model.pth'
        opt_file = model_folder + 'opt.pth'
        # Establish shape of the model based on data within file
        state_dict = torch.load(model_file, weights_only=True)
        self.nn_shape = {
            'players_input': state_dict['embedding_player.0.weight'].shape[1],
            'teams_input': state_dict['embedding_team.0.weight'].shape[1],
            'opps_input': state_dict['embedding_opp.0.weight'].shape[1],
            'embedding_player': state_dict['embedding_player.0.weight'].shape[0],
            'embedding_team': state_dict['embedding_team.0.weight'].shape[0],
            'embedding_opp': state_dict['embedding_opp.0.weight'].shape[0],
            'linear_stack': state_dict['linear_stack.0.weight'].shape[0],
            'stats_output': state_dict['linear_stack.2.weight'].shape[0],
        }
        self.nn_shape['stats_input'] = (state_dict['linear_stack.0.weight'].shape[1] -
                                self.nn_shape['embedding_player'] -
                                self.nn_shape['embedding_team'] -
                                self.nn_shape['embedding_opp']
                                )

        # Create Neural Network model with shape determined above
        self.model = NeuralNetwork(shape=self.nn_shape).to(self.device)

        # Load weights/biases into model
        self.model.load_state_dict(state_dict)

        # Initialize optimizer, then load state dict from file
        self.optimizer = torch.optim.SGD(self.model.parameters())
        opt_state_dict = torch.load(opt_file, weights_only=True)
        self.optimizer.load_state_dict(opt_state_dict)

        # Print model
        if print_loaded_model:
            logger.info(f'Loaded model from file {model_file}')
            self.print(self.model, log=True)


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


    def train_and_validate(self, train_dataloader, validation_dataloader, param_set=None, **kwargs):
        """Carries out the training process and generates predictions for a separate evaluation dataset.

            The model is first trained using train_dataloader and the input loss function (as well as 
            model and optimizer parameters contained in the NeuralNetPredictor object).
            After each training epoch, the validation data is used to generate prediction accuracy results.
            This process is repeated until either the max epochs condition or the end learning (model no longer
            improving) condition are met.

            Args:
                train_dataloader (DataLoader): data to use for Neural Net training.
                validation_dataloader (DataLoader): data to use for Neural Net validation.
                param_set (HyperParameterSet | dict, optional): set of hyper-parameters used in Neural Network training. 
                    If HyperParameterSet, may include:
                        loss_fn (HyperParameter): handle denoting the loss function to use (e.g. nn.MSELoss)
                    If dict, may include: 
                        loss_fn (function): handle denoting the loss function to use (e.g. nn.MSELoss)
                    Default values for each optional input:
                        loss_fn: nn.MSELoss

            Keyword-Args:
                All keyword arguments are passed to the eval_model method. See the related documentation for descriptions and valid inputs.
                All keyword arguments are optional.

            Returns:
                list: accuracy (quantified as average absolute prediction error) of the model's predictions on the
                    validation dataset after each epoch of training.
        """

        # Extract or set values of hyper-parameters
        if isinstance(param_set, HyperParameterSet):
            param_set = param_set.to_dict()
        if param_set is None:
            param_set = {}
        loss_fn = param_set.get('loss_fn', default_hp_values['loss_fn'])


        # Training/validation loop
        val_perfs = []
        for t in range(self.max_epochs):
            logger.info(f'Training Epoch {t+1}:')
            # Train
            self.__train(train_dataloader, loss_fn)

            # Validation
            val_result = self.eval_model(eval_dataloader=validation_dataloader, **kwargs)
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

            if print_losses and (num_batches < 10 or batch % int(num_batches / 10) == 0):
                loss, current = loss.item(), (batch + 1) * len(x_matrix)
                logger.debug(f'\tloss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
