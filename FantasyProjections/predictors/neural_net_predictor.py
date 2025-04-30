"""Creates and exports classes to be used as one approach to predicting NFL stats and Fantasy Football scores.

    Classes:
        NeuralNetPredictor : child of FantasyPredictor. Predicts NFL player stats using a Neural Net.
"""  # fmt: skip

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config.nn_config import default_nn_shape, nn_train_settings
from misc.manage_files import create_folders
from misc.stat_utils import stats_to_fantasy_points
from neural_net import HyperParameterSet, NeuralNetwork
from neural_net.nn_utils import compare_net_sizes
from predictors import FantasyPredictor

# Set up logger
logger = logging.getLogger("log")


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
            modify_hyper_parameter_values : Configures the neural net's hyper-parameters based on an input parameter set.
            configure_dataloader : Sets up DataLoader object for training with provided implementation details.
            configure_model_and_optimizer : Sets up NeuralNetwork and optimizer objects to train with provided implementation details.
            eval_model : Generates predicted stats for an input evaluation dataset, as computed by the NeuralNetwork.
            modify_hyper_parameter_values : Configures the neural net's hyper-parameters based on an input parameter set.
            load : Initializes a NeuralNetwork and optimizer using specifications saved to file.
            print : Displays the NeuralNetwork architecture and parameter size to console or to a logger.
            save : Stores NeuralNetwork and optimizer specifications to file.
            train_and_validate : Carries out the training process and generates predictions for a separate evaluation dataset.

    """  # fmt: skip

    # CONSTRUCTOR
    save_folder: str = None
    load_folder: str = None
    max_epochs: int = nn_train_settings["max_epochs"]
    n_epochs_to_stop: int = nn_train_settings["n_epochs_to_stop"]
    # hyper-parameters
    nn_shape: dict = None
    mini_batch_size: int = 0  # Will be set to a defined hyper-parameter value later
    learning_rate: float = 0  # Will be set to a defined hyper-parameter value later
    lmbda: float = 0  # Will be set to a defined hyper-parameter value later
    loss_fn = ""  # Will be set to a defined hyper-parameter value later

    def __post_init__(self):
        # Evaluates as part of the Constructor.
        # Generates attributes that are not simple data copies of inputs.

        # Assign default neural net shape (if shape not passed)
        if self.nn_shape is None:
            self.nn_shape = default_nn_shape

        # Assign a device
        self.device = self.__assign_device()
        # Initialize model
        if self.load_folder:
            self.load(self.load_folder, print_loaded_model=True)
        else:
            self.model = NeuralNetwork(shape=self.nn_shape).to(self.device)
            self.optimizer = torch.optim.SGD(self.model.parameters())

    # PUBLIC METHODS

    def configure_dataloader(self, dataset, **kwargs):
        """Sets up DataLoader object for training with provided implementation details.

            Args:
                dataset (StatsDataset): data to package into Dataloader
                kwargs:
                    shuffle (bool, optional): Whether to randomly shuffle the data in the dataset when creating the Dataloader. Defaults to False.
                    mini_batch (bool, optional): Whether to break the dataset into mini "batches" when creating the Dataloader (helpful for Neural Net training data).
                        If True, the batch size is determined by the mini_batch_size attribute of the NeuralNetPredictor object.
                        If False, the batch size is equivalent to the dataset size.
                        Defaults to False.

            Returns:
                DataLoader: Training DataLoader to use in training. Optionally batched/shuffled based on method inputs.

        """  # fmt: skip

        # Optional keyword arguments
        shuffle = kwargs.get("shuffle", False)
        mini_batch = kwargs.get("mini_batch", False)

        # Set batch size
        batch_size = int(self.mini_batch_size) if mini_batch else dataset.x_data.shape[0]

        # Create and return dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def configure_model_and_optimizer(self, **kwargs):
        """Sets up NeuralNetwork and optimizer objects to train with provided implementation details.

            Keyword-Args:
                print_model_flag (bool, optional): displays Neural Network model architecture to console or a logger.
                    Defaults to True.

            Side Effects (Modified Attributes):
                self.model is given a new instance of NeuralNetwork
                self.optimizer is given a new instance of torch.optim.SGD
                Note: both of these attributes are configured based on other attributes of the NeuralNetPredictor object. These include:
                    .nn_shape
                    .learning_rate
                    .lmbda
        """  # fmt: skip

        # Optional keyword arguments
        print_model_flag = kwargs.get("print_model_flag", True)

        self.model = NeuralNetwork(shape=self.nn_shape).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lmbda)

        if print_model_flag:
            self.print(self.model, log=True)

    def eval_model(self, eval_data=None, eval_dataloader=None, **kwargs):
        """Generates predicted stats for an input evaluation dataset, as computed by the NeuralNetwork.

            Either eval_data or eval_dataloader must be input.
            If both are input, eval_dataloader will be used.

            Args:
                eval_data (StatsDataset, optional): data to use for Neural Net evaluation (e.g. validation or test data). Defaults to None.
                eval_dataloader (DataLoader, optional): data to use for Neural Net evaluation (e.g. validation or test data). Defaults to None.
                kwargs:
                    All keyword arguments are passed to the function stats_to_fantasy_points and to the PredictionResult constructor.
                    See the related documentation for descriptions and valid inputs.
                    All keyword arguments are optional.

            Returns:
                PredictionResult: Object packaging the predicted and true stats together, which can be used for plotting,
                    performance assessments, etc.

        """  # fmt: skip

        # Raise error if normalized=False in kwargs
        if "normalized" in kwargs and not kwargs["normalized"]:
            msg = "NeuralNetPredictor evaluation data must be normalized."
            raise ValueError(msg)
        kwargs["normalized"] = True

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
            for x_matrix, y_vec in eval_dataloader:
                pred = torch.cat((pred, self.model(x_matrix)))
                y_matrix = torch.cat((y_matrix, y_vec))

        # Convert outputs into un-normalized statistics/fantasy points
        stat_predicts = stats_to_fantasy_points(pred, stat_indices=stat_columns, **kwargs)
        stat_truths = self.eval_truth(eval_dataloader.dataset, **kwargs)
        result = self._gen_prediction_result(stat_predicts, stat_truths, eval_dataloader.dataset, **kwargs)

        return result

    def load(self, model_folder, print_loaded_model=True):
        """Initializes a NeuralNetwork and optimizer using specifications saved to file.

            Assumes the file name for the model is "model.pth"
            And the file name for the optimizer is "opt.pth"

            Args:
                model_folder (str): path where "model.pth" and "optimizer.pth" are located
                print_loaded_model (bool, optional): displays Neural Network model architecture to console or a logger.
                    Defaults to True.

            Attributes modified:
                model (NeuralNetwork): Neural Network implemented via PyTorch.
                    Size/architecture and initial parameters determined by the loaded model.pth
                optimizer (torch.optim.SGD): Optimizer for model, implemented via PyTorch.
                    Parameters are determined by the loaded optimizer.pth
                nn_shape (dict): Neural Network layers and number of neurons per layer.

        """  # fmt: skip

        model_file = model_folder + "model.pth"
        opt_file = model_folder + "opt.pth"
        # Establish shape of the model based on data within file
        state_dict = torch.load(model_file, weights_only=True)
        self.nn_shape = {
            "players_input": state_dict["embedding_player.0.weight"].shape[1],
            "teams_input": state_dict["embedding_team.0.weight"].shape[1],
            "opps_input": state_dict["embedding_opp.0.weight"].shape[1],
            "embedding_player": state_dict["embedding_player.0.weight"].shape[0],
            "embedding_team": state_dict["embedding_team.0.weight"].shape[0],
            "embedding_opp": state_dict["embedding_opp.0.weight"].shape[0],
            "linear_stack": state_dict["linear_stack.0.weight"].shape[0],
            "stats_output": state_dict["linear_stack.2.weight"].shape[0],
        }
        self.nn_shape["stats_input"] = (
            state_dict["linear_stack.0.weight"].shape[1]
            - self.nn_shape["embedding_player"]
            - self.nn_shape["embedding_team"]
            - self.nn_shape["embedding_opp"]
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
            logger.info(f"Loaded model from file {model_file}")
            self.print(self.model, log=True)

    def modify_hyper_parameter_values(self, param_set):
        """Configures the neural net's hyper-parameters based on an input parameter set.

            Args:
                param_set (HyperParameterSet | dict, optional): set of hyper-parameters used in Neural Network training.
                    If HyperParameterSet, may include:
                        mini_batch_size (HyperParameter): size of training data batches used in SGD training iterations.
                        learning_rate (HyperParameter): Learning Rate used in SGD optimizer.
                        lmbda (HyperParameter): L2 Regularization Parameter used in SGD optimizer.
                        loss_fn (HyperParameter): handle denoting the loss function to use (e.g. nn.MSELoss)
                        {nn_shape key} (HyperParameter): any of the key names in the Neural Network shape dictionary, defining the number of neurons in that layer.
                    If dict, may include:
                        mini_batch_size (int): size of training data batches used in SGD training iterations.
                        learning_rate (float): Learning Rate used in SGD optimizer.
                        lmbda (float): L2 Regularization Parameter used in SGD optimizer.
                        loss_fn (function): handle denoting the loss function to use (e.g. nn.MSELoss)
                        {nn_shape key} (int): any of the key names in the Neural Network shape dictionary, defining the number of neurons in that layer.
                    Default values for each hyper-parameter are controlled in hp_config.py.

            Side Effects (Modified Attributes):
                Each hyper-parameter corresponds to a NeuralNetPredictor attribute with the same name*. If the hyper-parameter is
                modified in the param_set, this attribute will be modified to match.
                *The class attribute .nn_shape encompasses many hyper-parameters regarding the shape of the Network.

        """  # fmt: skip

        # Convert param_set to dictionary
        if isinstance(param_set, HyperParameterSet):
            param_set = param_set.to_dict()
        if param_set is None:
            param_set = {}

        # Extract values of hyper-parameters present in param_set
        self.mini_batch_size = int(param_set.get("mini_batch_size", self.mini_batch_size))
        self.learning_rate = param_set.get("learning_rate", self.learning_rate)
        self.lmbda = param_set.get("lmbda", self.lmbda)
        self.loss_fn = param_set.get("loss_fn", self.loss_fn)
        # Fold network-shape hyper-parameters into nn_shape attribute
        for shape_param in self.nn_shape:
            self.nn_shape[shape_param] = param_set.get(shape_param, self.nn_shape[shape_param])

    def print(self, model, log=False):
        """Displays the NeuralNetwork architecture and parameter size to console or to a logger.

            Args:
                model (NeuralNetwork): Neural Network implemented via PyTorch.
                log (bool, optional): True if output should be directed to a logger,
                    False if printed to console. Defaults to False.

        """  # fmt: skip

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        if log:
            logger.debug(model)
            logger.debug(f"Total tunable parameters: {total_params}")
        else:
            print()
            print(model)
            print(f"Total tunable parameters: {total_params}")

    def save(self):
        """Stores NeuralNetwork and optimizer specifications to file.

            The folder to use is specified by the NeuralNetPredictor's save_folder attribute.
            The NeuralNet model is always saved as "model.pth".
            The optimizer is always saved as "opt.pth"
        """  # fmt: skip

        # Check that folder exists, and set filenames
        create_folders(self.save_folder)
        model_save_file = self.save_folder + "model.pth"
        opt_save_file = self.save_folder + "opt.pth"
        # Save Neural Net model and optimizer
        torch.save(self.model.state_dict(), model_save_file)
        torch.save(self.optimizer.state_dict(), opt_save_file)
        logger.info(f"Saved PyTorch Model State to {model_save_file}")

    def train_and_validate(
        self,
        train_dataloader=None,
        validation_dataloader=None,
        training_data=None,
        validation_data=None,
        param_set=None,
        **kwargs,
    ):
        """Carries out the training process and generates predictions for a separate evaluation dataset.

            The model is first trained using train_dataloader and the input loss function (as well as
            model and optimizer parameters contained in the NeuralNetPredictor object).
            After each training epoch, the validation data is used to generate prediction accuracy results.
            This process is repeated until either the max epochs condition or the end learning (model no longer
            improving) condition are met.

            Args:
                train_dataloader (DataLoader): data to use for Neural Net training.
                validation_dataloader (DataLoader): data to use for Neural Net validation.
                training_data (StatsDataset): data to use for Neural Net training.
                validation_data (StatsDataset): data to use for Neural Net validation.
                param_set (HyperParameterSet | dict, optional): set of hyper-parameters used in Neural Network training.
                    If HyperParameterSet, may include:
                        loss_fn (HyperParameter): handle denoting the loss function to use (e.g. nn.MSELoss)
                    If dict, may include:
                        loss_fn (function): handle denoting the loss function to use (e.g. nn.MSELoss)
                kwargs:
                    All keyword arguments are passed to the eval_model method. See the related documentation for descriptions and valid inputs.
                    All keyword arguments are optional.

            Returns:
                list: accuracy (quantified as average absolute prediction error) of the model's predictions on the
                    validation dataset after each epoch of training.

        """  # fmt: skip

        # Set hyper-parameter values
        if param_set:
            self.modify_hyper_parameter_values(param_set)

        # Configure neural net and dataloaders for training
        if (train_dataloader is None) or (validation_dataloader is None):
            train_dataloader = self.configure_dataloader(training_data, mini_batch=True, shuffle=True)
            validation_dataloader = self.configure_dataloader(validation_data, mini_batch=False, shuffle=False)
        if not self.__model_and_optimizer_up_to_date():
            self.configure_model_and_optimizer()

        # Training/validation loop
        val_perfs = []
        for t in range(self.max_epochs):
            logger.info(f"Training Epoch {t + 1}:")
            # Train
            self.__train(train_dataloader, self.loss_fn)

            # Validation
            val_result = self.eval_model(eval_dataloader=validation_dataloader, **kwargs)
            val_perfs.append(np.mean(val_result.diff_pred_vs_truth(absolute=True)))

            # Check stopping condition
            if self.__end_learning(val_perfs, self.n_epochs_to_stop):  # Check stopping condition
                logger.info("Learning has stopped, terminating training process")
                break

        # Return final validation performance, and all validation performances
        return val_perfs[-1], val_perfs

    # PRIVATE METHODS

    def __assign_device(self, print_device=True):
        # Get cpu, gpu or mps device for training.
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if print_device:
            logger.info(f"Using {device} device")

        return device

    def __end_learning(self, perfs, n_epochs_to_stop, improvement_threshold=0.01):
        # Determines whether to stop training (if test performance has stagnated)
        # Returns true if learning should be stopped
        # If n_epochs_to_stop is less than zero, this feature is turned off
        # (always returns False)
        # Performance must improve by this factor in n_epochs_to_stop in order to
        # continue training

        return (
            len(perfs) > n_epochs_to_stop > 0 and perfs[-1] - perfs[-n_epochs_to_stop - 1] >= -improvement_threshold * perfs[-1]
        )

    def __model_and_optimizer_up_to_date(self):
        # Checks whether the properties of self.model are consistent with the shape in self.nn_shape
        # and the properties of self.optimizer are consistent with the NeuralNetPredictor's other attributes (like learning_rate)
        return compare_net_sizes(self.model, NeuralNetwork(shape=self.nn_shape)) and (
            self.optimizer == torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lmbda)
        )

    def __train(self, dataloader, loss_fn, print_losses=True):
        # Implements Stochastic Gradient Descent to train the model
        # against the provided training dataloader. Periodically logs losses from the loss function
        # Loss function may be a function handle or a subset of valid strings that match function handles

        # Handle optional string loss_fn
        loss_fn_strings = {"nn.MSELoss()": nn.MSELoss(), "nn.CrossEntropyLoss()": nn.CrossEntropyLoss()}
        if isinstance(loss_fn, str):
            try:
                loss_fn = loss_fn_strings[loss_fn]
            except KeyError as e:
                msg = "Invalid loss function string"
                raise KeyError(msg) from e

        size = len(dataloader.dataset)
        num_batches = int(np.ceil(size / dataloader.batch_size))
        self.model.train()
        for batch, (x_dataloader, y_dataloader) in enumerate(dataloader):
            x_matrix, y_matrix = x_dataloader.to(self.device), y_dataloader.to(self.device)

            # Compute prediction error
            pred = self.model(x_matrix)
            loss = loss_fn(pred, y_matrix)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            num_print_calls = 10
            batches_per_print_call = 10
            if print_losses and (num_batches < num_print_calls or batch % int(num_batches / batches_per_print_call) == 0):
                loss, current = loss.item(), (batch + 1) * len(x_matrix)
                logger.debug(f"\tloss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
