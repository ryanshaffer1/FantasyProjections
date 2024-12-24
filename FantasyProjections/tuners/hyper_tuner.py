"""Creates and exports multiple classes that support the handling and manipulation of Neural Network HyperParameters.
    Hyper-Parameter = Variable within ML equations which is not learned by the model during training, and must be set before training.

    This class cannot optimize HyperParameters on its own, but forms the building blocks and base class to implement optimization according to other algorithms.

    Classes:
        HyperParameterTuner : Base class for Tuner objects which modify values of a HyperParameterSet in order to optimize Neural Net performance according to some algorithm.
"""

import logging
import numpy as np
import pandas as pd
from misc.manage_files import create_folders

# Set up logger
logger = logging.getLogger('log')

class HyperParamTuner():
    """Base class for Tuner objects which modify values of a HyperParameterSet in order to optimize Neural Net performance according to some algorithm.
    
        Sub-classes implement specific prediction algorithms, including:
                GridSearchTuner: optimizes HyperParameters using a Recursive Grid Search algorithm.
    
        Attributes:
            param_set (HyperParameterSet): Set of HyperParameters to vary during optimization ("tuning") process.
            save_file (str, optional): path to file where any tuning performance logs should be saved. Defaults to None.
            optimize_hypers (bool, optional): Whether to vary the values of optimizable HyperParameters ("tune" the HyperParameters), or stick to the initial values provided.
                Defaults to False.
            plot_tuning_results (bool, optional): Whether to create a plot showing the performance for each iteration of HyperParameter tuning. Defaults to False.
            perf_list (list): Performance of Neural Net (e.g. Validation Error, loss, etc.) after each tuning iteration
            hyper_tuning_table (list): Table recording HyperParameter values and subsequent Neural Network performance after each tuning iteration

        Public Methods:
            None
    """

    def __init__(self, param_set, **kwargs):
        """Constructor for HyperParamTuner

            Args:
                param_set (HyperParameterSet): Set of HyperParameters to vary during optimization ("tuning") process.
                save_file (str, optional): path to file where any tuning performance logs should be saved. Defaults to None.
                optimize_hypers (bool, optional): Whether to vary the values of optimizable HyperParameters ("tune" the HyperParameters), or stick to the initial values provided.
                    Defaults to False.
                plot_tuning_results (bool, optional): Whether to create a plot showing the performance for each iteration of HyperParameter tuning. Defaults to False.
            
            Additional Class Attributes (generated, not passed as inputs):
                perf_list (list): Performance of Neural Net (e.g. Validation Error, loss, etc.) after each tuning iteration
                hyper_tuning_table (list): Table recording HyperParameter values and subsequent Neural Network performance after each tuning iteration
        """

        self.param_set = param_set.copy()
        self.save_file = kwargs.pop('save_file', None)
        self.optimize_hypers = kwargs.pop('optimize_hypers', False)
        self.plot_tuning_results = kwargs.pop('plot_tuning_results', False)

        # Initialize attributes to use later in tuning
        self.n_value_combinations = 0
        self.perf_list = [] # List of model performance values for all combos of hyperparameter values
        self.hyper_tuning_table = []


    # PUBLIC METHODS

    def refine_area_of_interest(self, ind):
        """Zooms in on an area of interest around a specified combination of hyper-parameter values.

            Args:
                ind (int): Index of HyperParameter.values attribute to refine grid around

            Modifies: 
                .val_range for each HyperParameter object in self.param_set.hyper_parameters
        """

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
                    hp.val_range = [center_val]


    def eval_hp_combinations(self, eval_function, save_function,
                              eval_kwargs, save_kwargs, **kwargs):
        """Evaluates a function for all combinations of HyperParameter values being considered, and tracks the optimal performance (min or max function output).
        
            Implements recursive grid search to optimize.

            Args:
                eval_function (function/method): Function to call in order to evaluate a model that uses the hyper-parameters and return an output value.
                    eval_function must take param_set as an input.
                save_function (function/method): Function to call whenever the best performance (so far) is achieved to save the model config.
                eval_kwargs (dict): Keyword-Arguments to pass to the evaluation function.
                save_kwargs (dict): Keyword-Arguments to pass to the save function.

            Keyword-Arguments:
                maximize (bool, optional): whether to maximize (True) or minimize (False) the values returned from eval_function. Defaults to False (minimize).

            Returns:
                float: Optimal performance across all function evaluations.
        """

        # Keyword arguments for this method
        maximize = kwargs.get('maximize', False)

        # Track optimal performance and index
        optimal_perf = 0 if maximize else np.inf
        optimal_ind = -1

        # Iterate through all combinations of hyperparameters
        for curr_ind in range(self.n_value_combinations):
            # Set and display hyperparameters for current run
            self.param_set.set_values(curr_ind)
            logger.info(f'HP Grid Point {curr_ind+1} of {self.n_value_combinations}: -------------------- ')
            self.param_set.print_values()

            # Function Evaluation
            result = eval_function(param_set=self.param_set, **eval_kwargs)
            eval_perf = result[0] if hasattr(result,'__iter__') else result

            # Track evaluation performance for the set of hyperparameters used
            self.perf_list.append(eval_perf)

            # Save the model if it is the best performing so far
            if (maximize and self.perf_list[curr_ind] > optimal_perf) or (not maximize and self.perf_list[curr_ind] < optimal_perf):
                optimal_ind = curr_ind
                optimal_perf = self.perf_list[curr_ind]
                if save_function is not None:
                    save_function(**save_kwargs)

        # Return optimal performance
        return optimal_ind, optimal_perf


    # PROTECTED METHODS

    def _save_hp_tuning_results(self, addl_columns=None, filename=None, log_name=None, optimal_ind=None):
        # Generates table with results of HyperParameter tuning (input HyperParameter values and output Neural Net performance),
        # and optionally saves the table to a file.

        # Handle additional columns input
        if not addl_columns:
            addl_columns = {}
        else:
            for key, val in addl_columns.items():
                addl_columns[key] = [val] if not hasattr(val,'__iter__') else val

        # Create array of all hyperparameter values in current tuning "batch"
        curr_results_table = []
        for hp in self.param_set.hyper_parameters:
            curr_results_table.append(hp.values)
        # Add performance of the model to the array
        curr_results_table.append(self.perf_list)
        # Add additional data to array
        for val in addl_columns.values():
            if len(val) == 1:
                curr_results_table.append(val*len(self.perf_list))
            elif len(val) == len(self.perf_list):
                curr_results_table.append(val)

        # Append the transpose of the curr_results_table to the results_array
        self.hyper_tuning_table = self.hyper_tuning_table + list(map(list, zip(*curr_results_table)))

        # Convert to dataframe (in order to add column labels)
        column_names = [hp.name for hp in self.param_set.hyper_parameters]
        column_names.extend(['Model Performance'] + list(addl_columns.keys()))
        hyper_tuning_df = pd.DataFrame(self.hyper_tuning_table,columns=column_names)

        # Optionally save to file
        if filename:
            create_folders(filename)
            hyper_tuning_df.to_csv(filename)

        # Optionally log optimal performance
        # Print out optimal performance
        if log_name is not None:
            logger.info(f'{log_name} Complete. Optimal performance: '
                f'{self.perf_list[optimal_ind]}. Hyper-parameters used: ')
            self.param_set.print_values()


        return hyper_tuning_df
