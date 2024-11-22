from dataclasses import dataclass, field, InitVar
import numpy as np
import pandas as pd

@dataclass
class HyperParameter():
    # CONSTRUCTOR
    name: str
    optimizable: InitVar[bool | None]
    value: float = 0
    values: list = field(init=False)
    val_range: list = None
    val_scale: str = 'none'
    num_steps: int = 1

    def __post_init__(self, optimizable):
        self.values = [self.value] # This may be overwritten by a HyperParameterSet object, if optimizing
        if not self.val_range or not optimizable:
            self.val_range = [self.value]
        self.gen_gridpoints()


    # PUBLIC METHODS

    def gen_gridpoints(self):
        if len(self.val_range) > 1:
            match self.val_scale:
                case 'linear':
                    self.gridpoints = np.interp(range(self.num_steps), [
                                                0, self.num_steps - 1], self.val_range)
                case 'log':
                    self.gridpoints = 10**np.interp(
                        range(self.num_steps), [0, self.num_steps - 1], np.log10(self.val_range))
                # Not as sure about the proper handling of these:
                case 'none':
                    self.val_range = self.value
                    self.gridpoints = self.val_range
                    self.num_steps = 1
                case 'selection':
                    self.gridpoints = self.val_range
                    self.num_steps = len(self.val_range)
                case _:
                    self.gridpoints = self.val_range
                    self.num_steps = 1
        else:
            self.gridpoints = [self.value]
            self.num_steps = 1


    def set_value(self, i):
        self.value = self.values[i]

@dataclass
class HyperParameterSet():
    hyper_parameters: tuple
    optimize: bool = True

    def __post_init__(self):
        if self.optimize:
            self.__gen_grid()
        else:
            self.total_gridpoints = 1
            for hp in self.hyper_parameters:
                hp.values = [hp.value]


    # PUBLIC METHODS

    def get(self,hp_name):
        # Returns the hyper-parameter in the set with the provided name.
        hp_names = [hp.name for hp in self.hyper_parameters]
        return self.hyper_parameters[hp_names.index(hp_name)]


    def refine_grid(self, ind):
        # "Zoom in" on the area of interest and generate new gridpoints closer to the provided index
        # Find new value ranges for each hyperparameter
        for hp in self.hyper_parameters:
            center_val = hp.values[ind]
            # gridpoints_ind = int(np.where(hp.gridpoints==center_val)[0])
            gridpoints_ind = list(hp.gridpoints).index(center_val)
            match hp.val_scale:
                case 'linear':
                    if gridpoints_ind == 0:
                        minval = center_val
                        maxval = (
                            hp.gridpoints[gridpoints_ind] + hp.gridpoints[gridpoints_ind + 1]) / 2
                    elif gridpoints_ind == hp.num_steps - 1:
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
                    elif gridpoints_ind == hp.num_steps - 1:
                        minval = 10**((np.log10(hp.gridpoints[gridpoints_ind - 1]) + np.log10(
                            hp.gridpoints[gridpoints_ind])) / 2)
                        maxval = center_val
                    else:
                        minval = 10**((np.log10(hp.gridpoints[gridpoints_ind - 1]) + np.log10(
                            hp.gridpoints[gridpoints_ind])) / 2)
                        maxval = 10**((np.log10(hp.gridpoints[gridpoints_ind]) + np.log10(
                            hp.gridpoints[gridpoints_ind + 1])) / 2)
                    hp.val_range = [minval, maxval]
                case 'selection':
                    # No setting a range, just select the value that performed
                    # best
                    hp.val_range = center_val
            # Generate new grid points for the hyperparameter based on its
            # "zoomed in" values of interest
            hp.gen_gridpoints()

        # After refining gridpoints, generate new array of points
        self.__gen_grid()


    def set_values(self, ind):
        if self.optimize:
            for hp in self.hyper_parameters:
                hp.set_value(ind)


    # PRIVATE METHODS

    def __gen_grid(self):
        # For each hyperparameter in the set, generates a list of all values to use in order.
        # The values are organized such that every index in the list represents a unique combination
        # of values across all the hyperparamters (ie a unique point on the
        # n-dimensional grid of hyperparameters)
        self.total_gridpoints = 1
        for ind, hp in enumerate(self.hyper_parameters):
            hp.values = np.repeat(hp.gridpoints, self.total_gridpoints)
            for prev in self.hyper_parameters[:ind]:
                prev.values = np.array(list(prev.values) * hp.num_steps)

            self.total_gridpoints *= hp.num_steps

@dataclass
class HyperParamTuner():
    param_set: HyperParameterSet
    save_file: str
    optimize_hypers: bool = False
    plot_tuning_results: bool = False

    def __post_init__(self):
        # Initialize attributes to use later in tuning
        self.model_perf_list = [] # List of model performance values for all combos of hyperparameter values
        self.hyper_tuning_table = []


    # PUBLIC METHODS

    # PROTECTED METHODS

    def _save_hp_tuning_results(self, addl_columns=None, filename=None):
        # Handle additional columns input
        if not addl_columns:
            addl_columns = {}
        else:
            for key, val in addl_columns.items():
                addl_columns[key] = [val] if not hasattr(val,'__iter__') else val

        # Create array of all hyperparameter values in current run ("layer") of grid search
        curr_results_table = []
        for hp in self.param_set.hyper_parameters:
            curr_results_table.append(hp.values)
        # Add performance of the model to the array
        curr_results_table.append(self.model_perf_list)
        # Add additional data to array
        for val in addl_columns.values():
            if len(val) == 1:
                curr_results_table.append(val*len(self.model_perf_list))
            elif len(val) == len(self.model_perf_list):
                curr_results_table.append(val)

        # Append the transpose of the curr_results_table to the results_array
        self.hyper_tuning_table = self.hyper_tuning_table + list(map(list, zip(*curr_results_table)))

        if filename:
            # Convert to dataframe (in order to add column labels) and save to file
            column_names = [hp.name for hp in self.param_set.hyper_parameters]
            column_names.extend(['Model Performance'] + list(addl_columns.keys()))
            pd.DataFrame(self.hyper_tuning_table,columns=column_names).to_csv(filename)
