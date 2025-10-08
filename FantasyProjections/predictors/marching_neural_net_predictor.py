"""Creates and exports classes to be used as one approach to predicting NFL stats and Fantasy Football scores.

    Classes:
        NeuralNetPredictor : child of FantasyPredictor. Predicts NFL player stats using a Neural Net.
"""  # fmt: skip

from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from misc.time_helper_functions import year_week_from_epweek
from predictors import NeuralNetPredictor

if TYPE_CHECKING:
    from misc.dataset import StatsDataset
    from neural_net import HyperParameterSet
    from results import PredictionResult


# Set up logger
logger = logging.getLogger("log")


@dataclass
class MarchingNeuralNetPredictor(NeuralNetPredictor):
    """Predictor of NFL players' stats in games, using a Neural Net to generate predictions.

        Sub-class of NeuralNetPredictor.

        Args:
            name (str): name of the predictor object, used for logging/display purposes.
            nn_shape (dict): Neural Network layers and number of neurons per layer.
            max_epochs (int): number of training iterations before stopping training.
            n_epochs_to_stop (int): number of training iterations to check for improvement before stopping.
                ex. if a NeuralNetPredictor's performance has not improved over its last n training epochs,
                the training process will be terminated.
            save_folder (str, optional): path to the folder to save model and optimizer settings. Defaults to None.
            load_folder (str, optional): path to the folder to load model and optimizer settings. Defaults to None.

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
    def __post_init__(self):
        super().__post_init__()
        # Enforce that self.save_folder is not None
        if self.save_folder is None:
            msg = "MarchingNeuralNetPredictor must have a save_folder specified."
            logger.error(msg)
            raise AttributeError(msg)

    # PUBLIC METHODS
    def eval_model(self, eval_data: StatsDataset, **kwargs) -> PredictionResult:
        """Generates predicted stats for an input evaluation dataset, based on training data from BEFORE the evaluation dataset.

            Loads a version of the predictor model which was generated based on training data from before the evaluation data.

            Args:
                eval_data (StatsDataset): data to use for Neural Net evaluation (e.g. validation or test data).
                kwargs:
                    All keyword arguments are passed to the function stats_to_fantasy_points and to the PredictionResult constructor.
                    See the related documentation for descriptions and valid inputs.
                    All keyword arguments are optional.

            Returns:
                PredictionResult: Object packaging the predicted and true stats together, which can be used for plotting,
                    performance assessments, etc.

        """  # fmt: skip

        # If eval data has no manager, just evaluate once as normal
        if eval_data.manager is None:
            result = super().eval_model(eval_data, **kwargs)
            return result

        # Loop through the manager's epweeks
        manager = eval_data.manager
        for i, epweek in enumerate(manager.all_epweeks):
            # Set datasets for this epweek
            datasets = manager.set_to_week(epweek)
            eval_data = datasets.get(eval_data.name, eval_data)

            # Find first week of the evaluation data
            eval_data_epweek = min(eval_data.id_data["EpWeek"])

            # Obtain a list of all epweeks with saved models (epweeks corresponding to the last training week)
            saved_epweeks = []
            for model_filename in glob.glob(os.path.join(self.save_folder, "model_*.pth")):  # type: ignore[reportArgumentType]
                model_epweek = re.match(r"model_(\d+).pth", os.path.basename(model_filename))
                if model_epweek:
                    saved_epweeks.append(int(model_epweek.group(1)))

            # Find the saved model that is based on training data most recent (but not after) the start of the evaluation data
            try:
                epweek_to_load = max(num for num in saved_epweeks if num < eval_data_epweek)
            except ValueError as e:
                msg = f'{self.name} has not been trained on a dataset ending before evaluation dataset "{eval_data.name}" start: {year_week_from_epweek(eval_data_epweek)}.'
                logger.exception(msg)
                raise ValueError(msg) from e

            # Load the model
            self.load(self.save_folder, epweek_to_load)  # type: ignore[reportArgumentType]

            # Evaluate the model
            weekly_result = super().eval_model(eval_data, **kwargs)

            # Append results from each week's evaluation into an overall PredictionResult
            result = result.append(weekly_result) if i > 0 else weekly_result

        return result

    def load(self, model_folder: str, epweek: int, print_loaded_model: bool = False) -> None:
        """Initializes a NeuralNetwork and optimizer using specifications saved to file.

            Assumes the file name for the model is "model_<epweek>.pth"
            And the file name for the optimizer is "opt.pth"

            Args:
                model_folder (str): path where "model.pth" and "optimizer.pth" are located
                epweek (int): current epoch week, used to label the model file.
                print_loaded_model (bool, optional): displays Neural Network model architecture to console or a logger.
                    Defaults to False.

            Attributes modified:
                model (NeuralNetwork): Neural Network implemented via PyTorch.
                    Size/architecture and initial parameters determined by the loaded model.pth
                optimizer (torch.optim.SGD): Optimizer for model, implemented via PyTorch.
                    Parameters are determined by the loaded optimizer.pth
                nn_shape (dict): Neural Network layers and number of neurons per layer.

        """  # fmt: skip

        model_file = f"model_{epweek}.pth"
        super().load(model_folder, model_file=model_file, print_loaded_model=print_loaded_model)

    def save(self, epweek: int) -> None:
        """Stores NeuralNetwork and optimizer specifications to file.

            The folder to use is specified by the NeuralNetPredictor's save_folder attribute.
            The NeuralNet model is always saved as "model_<epweek>.pth".
            The optimizer is always saved as "opt.pth"

            Args:
                epweek (int): current epoch week, used to label the model file.

        """  # fmt: skip

        model_file = f"model_{epweek}.pth"
        super().save(model_file=model_file)

    def manage_training_and_validation(
        self,
        training_data: StatsDataset,
        validation_data: StatsDataset,
        param_set: HyperParameterSet | dict | None = None,
        **kwargs,
    ) -> float:
        # Check both required datasets are provided
        if training_data is None or validation_data is None:
            msg = f"{self.name}, manage_training_and_validation: missing training or validation data"
            logger.exception(msg)
            raise ValueError(msg)

        # Check if either dataset has a manager, and use that for training/validation (if both have managers, use the training data manager)
        manager = training_data.manager if training_data.manager is not None else validation_data.manager

        # If neither dataset has a manager, train/validate once as normal
        if manager is None:
            val_perf, _ = self.train_and_validate(
                training_data=training_data,
                validation_data=validation_data,
                param_set=param_set,
                **kwargs,
            )
            return val_perf

        # Loop through the manager's epweeks
        val_perfs = []
        for i, epweek in enumerate(manager.all_epweeks):
            logger.info(f"Training Week {i + 1} of {len(manager.all_epweeks)}: {year_week_from_epweek(epweek)}")
            # Set datasets for this epweek
            datasets = manager.set_to_week(epweek)
            training_data = datasets.get(training_data.name, training_data)
            validation_data = datasets.get(validation_data.name, validation_data)

            # Perform training/validation for this epweek
            val_perf, _ = self.train_and_validate(
                training_data=training_data,
                validation_data=validation_data,
                param_set=param_set,
                **kwargs,
            )
            val_perfs.append(val_perf)

            # Save model after each week's training, using the last week of the training dataset
            last_training_epweek = max(training_data.id_data["EpWeek"])
            if self.save_folder is not None:
                self.save(last_training_epweek)

        # Average validation performance over all weeks is the figure of merit
        val_perf = float(np.mean(val_perfs))

        return val_perf
