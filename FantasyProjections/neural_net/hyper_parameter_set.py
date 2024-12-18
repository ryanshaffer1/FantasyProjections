"""Creates and exports a class that supports the handling and manipulation of Neural Network HyperParameters.
    Hyper-Parameter = Variable within ML equations which is not learned by the model during training, and must be set before training.

    Classes:
        HyperParameterSet : Groups HyperParameter objects together and allows for simultaneous modification of multiple HyperParameters.
"""

import numpy as np
from . import HyperParameter

class HyperParameterSet():
    """Groups HyperParameter objects together and allows for simultaneous modification of multiple HyperParameters.
    
        Attributes:
            hyper_parameters (tuple): tuple of HyperParameter objects contained within the HyperParameterSet.
            optimize (bool): Whether to vary the values of optimizable HyperParameters ("tune" the HyperParameters), or stick to the initial values provided.
            total_gridpoints (int): Number of unique combinations of HyperParameter values.

        Public Methods:
            get : Returns a HyperParameter from a HyperParameterSet based on the HyperParameter's name.
            refine_grid : Implements Recursive Grid Search by generating new gridpoints/values for each HyperParameter, "zooming in" closer to the values at the provided index.
            set_values : Sets value of all HyperParameter objects in set to value at a specific index within the list of values.
            to_dict : Converts object data to a dict, where each key is the name of a HyperParameter, and each value is the HyperParameter's current value.
    """
    def __init__(self,
                 hp_set: tuple = None,
                 hp_dict: dict = None,
                 **kwargs):
        """Constructor for HyperParameterSet.

            One of the following initialization options must be followed. If both inputs are passed, hyper_parameters takes precedence.

            Args (Initialization Option 1):
                hp_set (tuple, optional): tuple of pre-initialized HyperParameter objects.
            Args (Initialization Option 2):
                hp_dict (dict, optional): dictionary containing names of each HyperParameter as keys, mapping to a dictionary 
                    listing all attributes for that HyperParameter (any attributes not set will be assigned the default values in HyperParameter init).

            Keyword-Args: 
                optimize (bool, optional): Whether to vary the values of optimizable HyperParameters ("tune" the HyperParameters), 
                    or stick to the initial values provided. Defaults to True.
        """
        # Optional keyword arguments
        self.optimize = kwargs.get('optimize', True)

        if hp_set is not None:
            self.hyper_parameters = hp_set
        else:
            if hp_dict is None:
                raise ValueError('Input must be provided to initialize HyperParameterSet!')

            self.__gen_hp_set(hp_dict)

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
                    hp.val_range = [float(minval), float(maxval)]
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


    def to_dict(self):
        """Converts object data to a dict, where each key is the name of a HyperParameter, and each value is the HyperParameter's current value.

            Returns:
                dict: dictionary mapping HyperParameters in the set to their current values.
        """
        hp_dict = {hp.name:hp.value for hp in self.hyper_parameters}
        return hp_dict

    # PRIVATE METHODS

    def __gen_grid(self):
        # For each hyperparameter in the set, generates a list of all values to use in order.
        # The values are organized such that every index in the list represents a unique combination
        # of values across all the hyperparamters (ie a unique point on the
        # n-dimensional grid of hyperparameters)
        self.total_gridpoints = 1
        for ind, hp in enumerate(self.hyper_parameters):
            hp.values = np.repeat(hp.gridpoints, self.total_gridpoints).tolist()
            for prev in self.hyper_parameters[:ind]:
                prev.values = np.array(list(prev.values) * hp.num_steps).tolist()

            self.total_gridpoints *= hp.num_steps

    def __gen_hp_set(self, hp_dict):
        # Converts a dict of dicts (where each key is the name of a HyperParameter, and each value is the attributes for that HyperParameter)
        # into a list of HyperParameter objects.
        hp_set = []

        for key, values in hp_dict.items():
            hyper_parameter = HyperParameter(name=key, **values)
            hp_set.append(hyper_parameter)

        self.hyper_parameters = tuple(hp_set)
