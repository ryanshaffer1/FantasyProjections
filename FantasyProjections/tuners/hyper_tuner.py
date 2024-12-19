"""Creates and exports multiple classes that support the handling and manipulation of Neural Network HyperParameters.
    Hyper-Parameter = Variable within ML equations which is not learned by the model during training, and must be set before training.

    This class cannot optimize HyperParameters on its own, but forms the building blocks and base class to implement optimization according to other algorithms.

    Classes:
        HyperParameterTuner : Base class for Tuner objects which modify values of a HyperParameterSet in order to optimize Neural Net performance according to some algorithm.
"""

from dataclasses import dataclass
import pandas as pd
from neural_net import HyperParameterSet

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
            perf_list (list): Performance of Neural Net (e.g. Validation Error, loss, etc.) after each tuning iteration
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
        self.perf_list = [] # List of model performance values for all combos of hyperparameter values
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
            hyper_tuning_df.to_csv(filename)

        return hyper_tuning_df
