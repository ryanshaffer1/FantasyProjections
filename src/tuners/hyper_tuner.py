"""Creates and exports multiple classes that support the handling and manipulation of Neural Network HyperParameters.
    Hyper-Parameter = Variable within ML equations which is not learned by the model during training, and must be set before training.

    This class cannot optimize HyperParameters on its own, but forms the building blocks and base class to implement optimization according to other algorithms.

    Classes:
        HyperParameter : Class handling the value and variations of a Neural Net Hyper-Parameter.
        HyperParameterSet : Groups HyperParameter objects together and allows for simultaneous modification of multiple HyperParameters.
        HyperParameterTuner : Base class for Tuner objects which modify values of a HyperParameterSet in order to optimize Neural Net performance according to some algorithm.
"""

from dataclasses import dataclass, field, InitVar
import numpy as np
import pandas as pd

@dataclass
class HyperParameter():
    """Class handling the value and variations of a Neural Net Hyper-Parameter.
    
        Hyper-Parameter = Variable within ML equations which is not learned by the model during training, and must be set before training.
    
        Args:
            name (str): name of the hyper-parameter. Used within training logic and must match a set list of valid hyper-parameters:
                - learning_rate
                - lmbda
                - loss_fn
                - mini_batch_size
            optimizable (bool): Whether the hyper-parameter can be modified by a HyperParameter Tuner. (Whether it is actually optimized depends on the tuner.)
                Note that optimizable is not tracked as an object attribute, being redundant to other data.
            value (float, optional): Initial value to assume if not optimizing. Defaults to 0.
            val_range (list, optional): Minimum and maximum values to use if optimizing. Defaults to [self.value] (preventing optimizing).
            val_scale (str, optional): Type of scale to use when setting values within val_range. Defaults to "none". Options include:
                - "none"
                - "linear"
                - "log"
                - "selection"
            num_steps (int, optional): Number of unique values to use when optimizing. Defaults to 1 (preventing optimizing).
        
        Additional Class Attributes:
            values (list): Sequence of values to use as the value attribute over successive tuning iterations. Determined by a HyperParameterTuner.
                Unique values may repeat.
            gridpoints (list): Array of unique values (with no repetition) to use in a grid search HyperParameter optimization.

        Public Methods:
            gen_gridpoints : Generates list of unique values to use in a grid search HyperParamater optimization.
            set_value : Sets HyperParameter attribute value to a specific index within the list of values.
    """

    # CONSTRUCTOR
    name: str
    optimizable: InitVar[bool | None]
    value: float = 0
    values: list = field(init=False)
    val_range: list = None
    val_scale: str = 'none'
    num_steps: int = 1

    def __post_init__(self, optimizable):
        # Evaluates as part of the Constructor.
        # Generates attributes that are not simple data copies of inputs.

        self.values = [self.value] # This may be overwritten by a HyperParameterSet object, if optimizing
        if not self.val_range or not optimizable:
            self.val_range = [self.value]
        self.gen_gridpoints()


    # PUBLIC METHODS

    def gen_gridpoints(self):
        """Generates list of unique values to use in a grid search HyperParamater optimization.
        
            Uses object attributes val_range, val_scale, and num_steps. 
            Creates object attribute gridpoints. May update num_steps to 1 if necessary.        
        """

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
        """Sets HyperParameter attribute value to a specific index within the list of values (list determined by a HyperParameterTuner).

            Args:
                i (int): Index of object attribute values to use as new self.value.
        """

        self.value = self.values[i]


@dataclass
class HyperParameterSet():
    """Groups HyperParameter objects together and allows for simultaneous modification of multiple HyperParameters.
    
        Args:
            hyper_parameters (tuple): tuple of HyperParameter objects. All HyperParameters must be initialized prior to initializing HyperParameterSet.
            optimize (bool): Whether to vary the values of optimizable HyperParameters ("tune" the HyperParameters), or stick to the initial values provided.
        
        Additional Class Attributes:
            total_gridpoints (int): Number of unique combinations of HyperParameter values.

        Public Methods:
            get : Returns a HyperParameter from a HyperParameterSet based on the HyperParameter's name.
            refine_grid : Implements Recursive Grid Search by generating new gridpoints/values for each HyperParameter, "zooming in" closer to the values at the provided index.
            set_values : Sets value of all HyperParameter objects in set to value at a specific index within the list of values.
    """

    hyper_parameters: tuple
    optimize: bool = True

    def __post_init__(self):
        # Evaluates as part of the Constructor.
        # Generates attributes that are not simple data copies of inputs.

        if self.optimize:
            self.__gen_grid()
        else:
            self.total_gridpoints = 1
            for hp in self.hyper_parameters:
                hp.values = [hp.value]


    # PUBLIC METHODS

    def get(self,hp_name):
        """Returns a HyperParameter from a HyperParameterSet based on the HyperParameter's name.

            Args:
                hp_name (str): Name of HyperParameter object within HyperParameterSet to return.

            Returns:
                HyperParameter: (first) object in HyperParameterSet.hyper_parameters with a name matching the input.
        """

        # Returns the hyper-parameter in the set with the provided name.
        hp_names = [hp.name for hp in self.hyper_parameters]
        return self.hyper_parameters[hp_names.index(hp_name)]


    def refine_grid(self, ind):
        """Implements Recursive Grid Search by generating new gridpoints/values for each HyperParameter, "zooming in" closer to the values at the provided index. 

            Args:
                ind (int): Index of HyperParameter.values attribute to refine grid around
            Modifies: 
                .gridpoints and .values for each object in self.hyper_parameters
        """

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
        """Sets value of all HyperParameter objects in set to value at a specific index within the list of values.

            Args:
                ind (int): Index of object attribute values to use as new self.value.
        """

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
    """Base class for Tuner objects which modify values of a HyperParameterSet in order to optimize Neural Net performance according to some algorithm.
    
        Sub-classes implement specific prediction algorithms, including:
                GridSearchTuner: optimizes HyperParameters using a Recursive Grid Search algorithm.
    
        Args:
            param_set (HyperParameterSet): Set of HyperParameters to vary during optimization ("tuning") process.
            save_folder (str): path to folder where any tuning performance logs should be saved.
            optimize_hypers (bool, optional): Whether to vary the values of optimizable HyperParameters ("tune" the HyperParameters), or stick to the initial values provided.
                Defaults to False.
            plot_tuning_results (bool, optional): Whether to create a plot showing the performance for each iteration of HyperParameter tuning. Defaults to False.
        
        Additional Class Attributes:
            model_perf_list (list): Performance of Neural Net (e.g. Validation Error, loss, etc.) after each tuning iteration
            hyper_tuning_table (list): Table recording HyperParameter values and subsequent Neural Network performance after each tuning iteration

        Public Methods:
            None
    """


    param_set: HyperParameterSet
    save_folder: str
    optimize_hypers: bool = False
    plot_tuning_results: bool = False

    def __post_init__(self):
        # Evaluates as part of the Constructor.
        # Generates attributes that are not simple data copies of inputs.

        # Initialize attributes to use later in tuning
        self.model_perf_list = [] # List of model performance values for all combos of hyperparameter values
        self.hyper_tuning_table = []


    # PUBLIC METHODS

    # PROTECTED METHODS

    def _save_hp_tuning_results(self, addl_columns=None, filename=None):
        # Generates table with results of HyperParameter tuning (input HyperParameter values and output Neural Net performance),
        # and optionally saves the table to a file.

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
