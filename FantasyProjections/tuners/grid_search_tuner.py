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
        
        Additional Class Attributes:
            save_file (str): path to file where tuning performance log will be saved. Filename is "hyper_grid.csv".
            total_cominations (int): Number of unique combinations of HyperParameter values.

        Adds Public Attributes to Other Classes:
            HyperParameter objects within param_set:
                gridpoints (list): Array of unique values (with no repetition) to use in a grid search HyperParameter optimization.

        Public Methods:
            refine_grid : Implements Recursive Grid Search by generating new gridpoints/values for each HyperParameter, "zooming in" closer to the values at the provided index.
            tune_hyper_parameters : Performs iterative evaluation of a function that depends on HyperParameters to find an optimal combination of HyperParameters.
    """

    hyper_tuner_layers: int = 1
    hyper_tuner_steps_per_dim: int = 3

    def __post_init__(self):
        super().__post_init__()
        self.save_file = self.save_folder + 'hyper_grid.csv'

        # Generate initial grid points
        if self.optimize_hypers:
            for hp in self.param_set.hyper_parameters:
                self.__gen_gridpoints(hp)
            self.__gen_hp_value_combos()
        else:
            # Adjust variables if not optimizing hyper-parameters
            self.hyper_tuner_layers = 1
            self.total_combinations = 1

    # PUBLIC METHODS

    def refine_grid(self, ind):
        """Implements Recursive Grid Search by generating new gridpoints/values for each HyperParameter, "zooming in" closer to the values at the provided index. 

            Args:
                ind (int): Index of HyperParameter.values attribute to refine grid around
            Modifies: 
                .gridpoints and .param_set.values for each HyperParameter object in self.param_set.hyper_parameters
        """

        # "Zoom in" on the area of interest and generate new gridpoints closer to the provided index
        # Find new value ranges for each hyperparameter
        for hp in self.param_set.hyper_parameters:
            center_val = hp.values[ind]
            gridpoints_ind = list(hp.gridpoints).index(center_val)
            match hp.val_scale:
                case 'linear':
                    if gridpoints_ind == 0:
                        minval = center_val
                        maxval = (
                            hp.gridpoints[gridpoints_ind] + hp.gridpoints[gridpoints_ind + 1]) / 2
                    elif gridpoints_ind == len(hp.gridpoints) - 1:
                        minval = (
                            hp.gridpoints[gridpoints_ind - 1] + hp.gridpoints[gridpoints_ind]) / 2
                        maxval = center_val
                    else:
                        minval = (
                            hp.gridpoints[gridpoints_ind - 1] + hp.gridpoints[gridpoints_ind]) / 2
                        maxval = (
                            hp.gridpoints[gridpoints_ind] + hp.gridpoints[gridpoints_ind + 1]) / 2
                    hp.val_range = [minval, maxval]
                case 'log':
                    if gridpoints_ind == 0:
                        minval = center_val
                        maxval = 10**((np.log10(hp.gridpoints[gridpoints_ind]) + np.log10(
                            hp.gridpoints[gridpoints_ind + 1])) / 2)
                    elif gridpoints_ind == len(hp.gridpoints) - 1:
                        minval = 10**((np.log10(hp.gridpoints[gridpoints_ind - 1]) + np.log10(
                            hp.gridpoints[gridpoints_ind])) / 2)
                        maxval = center_val
                    else:
                        minval = 10**((np.log10(hp.gridpoints[gridpoints_ind - 1]) + np.log10(
                            hp.gridpoints[gridpoints_ind])) / 2)
                        maxval = 10**((np.log10(hp.gridpoints[gridpoints_ind]) + np.log10(
                            hp.gridpoints[gridpoints_ind + 1])) / 2)
                    hp.val_range = [float(minval), float(maxval)]
                case 'selection':
                    # No setting a range, just select the value that performed
                    # best
                    hp.val_range = center_val
            # Generate new grid points for the hyperparameter based on its
            # "zoomed in" values of interest
            self.__gen_gridpoints(hp)

        # After refining gridpoints, generate new array of points
        self.__gen_hp_value_combos()


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

            Returns:
                float: Optimal performance across all function evaluations.
        """

        # Handle keyword arguments for each called function
        [eval_kwargs, save_kwargs, reset_kwargs] = [{} if kwargs is None else kwargs for kwargs in [eval_kwargs, save_kwargs, reset_kwargs]]

        # Keyword arguments for this method
        maximize = kwargs.get('maximize', False)

        for tune_layer in range(self.hyper_tuner_layers):
            logger.info(f'Optimization Round {tune_layer+1} of {self.hyper_tuner_layers} -------------------------------')
            # Iterate through all combinations of hyperparameters
            for grid_ind in range(self.total_combinations):
                # Set and display hyperparameters for current run
                self.param_set.set_values(grid_ind)
                logger.info(f'HP Grid Point {grid_ind+1} of {self.total_combinations}: -------------------- ')
                for hp in self.param_set.hyper_parameters:
                    logger.info(f"\t{hp.name} = {hp.value}")

                # Function Evaluation
                result = eval_function(param_set=self.param_set, **eval_kwargs)
                eval_perf = result[0] if hasattr(result,'__iter__') else result

                # Track evaluation performance for the set of hyperparameters used
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

        # Return optimal performance
        return self.perf_list[optimal_ind]

    # PRIVATE METHODS


    def __gen_hp_value_combos(self):
        # For each hyperparameter in the set, generates a list of all values to use in order.
        # The values are organized such that every index in the list represents a unique combination
        # of values across all the hyperparamters (ie a unique point on the
        # n-dimensional grid of hyperparameters)
        self.total_combinations = 1
        for ind, hp in enumerate(self.param_set.hyper_parameters):
            hp.values = np.repeat(hp.gridpoints, self.total_combinations).tolist()
            for prev in self.param_set.hyper_parameters[:ind]:
                prev.values = np.array(list(prev.values) * len(hp.gridpoints)).tolist()

            self.total_combinations *= len(hp.gridpoints)


    def __gen_gridpoints(self, hp):
        """Generates list of unique values to use in a grid search HyperParamater optimization.
        
            Args:
                hp (HyperParameter): HyperParameter object with a modifiable value and optional ranges, optimizability, etc.
                Uses object attributes val_range and val_scale. 

            Side Effects (Modified Attributes):
                hp:
                    Creates object attribute gridpoints.
                    gridpoints (list): Array of unique values (with no repetition) to use in a grid search HyperParameter optimization.
        """

        if isinstance(hp.val_range,list):
            match hp.val_scale:
                case 'linear':
                    hp.gridpoints = np.interp(range(self.hyper_tuner_steps_per_dim), [
                                                0, self.hyper_tuner_steps_per_dim - 1], hp.val_range).tolist()
                case 'log':
                    hp.gridpoints = (10**np.interp(
                        range(self.hyper_tuner_steps_per_dim), [0, self.hyper_tuner_steps_per_dim - 1], np.log10(hp.val_range))).tolist()
                # Not as sure about the proper handling of these:
                case 'none':
                    hp.val_range = [hp.value]
                    hp.gridpoints = hp.val_range
                case 'selection':
                    hp.gridpoints = hp.val_range
                case _:
                    hp.gridpoints = hp.val_range
        else:
            hp.gridpoints = [hp.value]


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
            self.refine_grid(optimal_ind)
            self.perf_list = []
            logger.info('Beginning next hyper-parameter optimization iteration.')
        else:
            logger.info('Model Training Complete!')
