"""Creates and exports class to be used as one approach to optimizing HyperParameters for a Neural Network.

    Classes:
        GridSerachTuner : Optimizes HyperParameters of a Neural Network for best performance (minimum evaluation error after training) via a Recursive Grid Search algorithm.
            Child of HyperParameterTuner.
"""

from dataclasses import dataclass
import logging
import numpy as np
from misc.manage_files import create_folders
from .hyper_tuner import HyperParamTuner
from .plot_grid_search_results import plot_grid_search_results

# Set up logger
logger = logging.getLogger('log')

@dataclass
class GridSearchTuner(HyperParamTuner):
    """Optimizes HyperParameters of a Neural Network for best performance (minimum evaluation error after training) via a Recursive Grid Search algorithm.
    
        Sub-class of HyperParameterTuner.
    
        Args:
            param_set (HyperParameterSet): Set of HyperParameters to vary during optimization ("tuning") process.
            save_folder (str): path to folder where any tuning performance logs should be saved.
            optimize_hypers (bool, optional): Whether to vary the values of optimizable HyperParameters ("tune" the HyperParameters), or stick to the initial values provided.
                Defaults to False.
            plot_tuning_results (bool, optional): Whether to create a plot showing the performance for each iteration of HyperParameter tuning. Defaults to False.
            hyper_tuner_layers (int, optional): Number of grid search layers to perform (each recursion layer is "zooming in" closer to a local optima.) Defaults to 1 (no zooming in).
            hyper_tuner_steps_per_dim (int, optional): Number of unique values to use for each optimizable HyperParameter. Defaults to 3.
            scale_epochs_over_layers (bool, optional): Whether each grid search layer gets a doubling of max epoch and early stopping condition. Defaults to False.
        
        Additional Class Attributes:
            save_folder (str): path to file where tuning performance log will be saved. Filename is "hyper_grid.csv".

        Public Methods:
            tune_hyper_parameters : Performs iterative evaluation of a function that depends on HyperParameters to find an optimal combination of HyperParameters.
    """

    hyper_tuner_layers: int = 1
    hyper_tuner_steps_per_dim: int = 3
    scale_epochs_over_layers: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.save_file = self.save_folder + 'hyper_grid.csv'
        # Adjust variables if not optimizing hyper-parameters
        if not self.optimize_hypers:
            self.hyper_tuner_layers = 1


    # PUBLIC METHODS

    def tune_hyper_parameters(self, eval_function, save_function=None, reset_function=None,
                              eval_kwargs=None, save_kwargs=None, reset_kwargs=None, **kwargs):
        """Performs iterative evaluation of a function that depends on HyperParameters to find an optimal combination of HyperParameters. 
        
            Implements recursive grid search to optimize.

            Args:
                eval_function (function/method): Function to call in order to evaluate a model that uses the hyper-parameters and return an output value.
                    eval_function must take param_set as an input.
                save_function (function/method, optional): Function to call whenever the best performance (so far) is achieved to save the model config.
                    Defaults to None.
                reset_function (function/method, optional): Function to call when tuning is done in order to reset the model to the best config. 
                    Defaults to None.
                eval_kwargs (dict, optional): Keyword-Arguments to pass to the evaluation function. Defaults to None.
                save_kwargs (dict, optional): Keyword-Arguments to pass to the save function. Defaults to None.
                reset_kwargs (dict, optional): Keyword-Arguments to pass to the reset function. Defaults to None.

            Keyword-Arguments:
                maximize (bool, optional): whether to maximize (True) or minimize (False) the values returned from eval_function. Defaults to False (minimize).

        """

        # Handle keyword arguments for each called function
        [eval_kwargs, save_kwargs, reset_kwargs] = [{} if kwargs is None else kwargs for kwargs in [eval_kwargs, save_kwargs, reset_kwargs]]

        # Keyword arguments for this method
        maximize = kwargs.get('maximize', False)

        for tune_layer in range(self.hyper_tuner_layers):
            logger.info(f'Optimization Round {tune_layer+1} of {self.hyper_tuner_layers} -------------------------------')
            # Iterate through all combinations of hyperparameters
            for grid_ind in range(self.param_set.total_gridpoints):
                # Set and display hyperparameters for current run
                self.param_set.set_values(grid_ind)
                logger.info(f'HP Grid Point {grid_ind+1} of {self.param_set.total_gridpoints}: -------------------- ')
                for hp in self.param_set.hyper_parameters:
                    logger.info(f"\t{hp.name} = {hp.value}")

                # ---------------------
                # Model Training and Validation Testing
                # ---------------------
                result = eval_function(param_set=self.param_set, **eval_kwargs)
                eval_perf = result[0] if hasattr(result,'__iter__') else result

                # Track validation performance for the set of hyperparameters used
                self.perf_list.append(eval_perf)
                # Save the model if it is the best performing so far
                optimal_ind = np.nanargmax(self.perf_list) if maximize else np.nanargmin(self.perf_list)
                if grid_ind == optimal_ind:
                    if save_function is not None:
                        save_function(**save_kwargs)

            # Print some results, determine whether to perform another layer of grid search, and if so, refine the mesh
            self.__next_hp_layer(tune_layer, optimal_ind)

        # After grid search finishes, plot results
        if len(self.hyper_tuning_table) > 0 and self.plot_tuning_results:
            plot_grid_search_results(self.save_file,self.param_set,variables=('learning_rate','lmbda'))

        # Set the model back to the highest performing config
        if reset_function is not None:
            reset_function(**reset_kwargs)


    # PRIVATE METHODS

    def __next_hp_layer(self,tune_layer, optimal_ind):
        # Generates the grid for the next layer of a recursive grid search.

        if self.optimize_hypers:
            # Save the results of the previous layer
            create_folders(self.save_folder)
            self._save_hp_tuning_results(addl_columns={'Grid Search Layer': tune_layer}, filename=self.save_file)
            # Print out optimal performance
            logger.info(
                f'Layer {tune_layer+1} '
                f'Complete. Optimal performance: '
                f'{self.perf_list[optimal_ind]}. '
                f'Hyper-parameters used: '
                )
            for hp in self.param_set.hyper_parameters:
                logger.info(f"\t{hp.name} = {hp.values[optimal_ind]}")

        if tune_layer < self.hyper_tuner_layers-1:
            self.param_set.refine_grid(optimal_ind)
            self.perf_list = []
            logger.info('Beginning next hyper-parameter optimization iteration.')
        else:
            logger.info('Model Training Complete!')
