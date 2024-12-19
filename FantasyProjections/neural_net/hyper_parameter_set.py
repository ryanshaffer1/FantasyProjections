"""Creates and exports a class that supports the handling and manipulation of Neural Network HyperParameters.
    Hyper-Parameter = Variable within ML equations which is not learned by the model during training, and must be set before training.

    Classes:
        HyperParameterSet : Groups HyperParameter objects together and allows for simultaneous modification of multiple HyperParameters.
"""

from . import HyperParameter

class HyperParameterSet():
    """Groups HyperParameter objects together and allows for simultaneous modification of multiple HyperParameters.
    
        Attributes:
            hyper_parameters (tuple): tuple of HyperParameter objects contained within the HyperParameterSet.

        Public Methods:
            get : Returns a HyperParameter from a HyperParameterSet based on the HyperParameter's name.
            set_values : Sets value of all HyperParameter objects in set to value at a specific index within the list of values.
            to_dict : Converts object data to a dict, where each key is the name of a HyperParameter, and each value is the HyperParameter's current value.
    """
    def __init__(self, hp_set: tuple = None, hp_dict: dict = None):
        """Constructor for HyperParameterSet.

            One of the following initialization options must be followed. If both inputs are passed, hyper_parameters takes precedence.

            Args (Initialization Option 1):
                hp_set (tuple, optional): tuple of pre-initialized HyperParameter objects.
                    Note: a singular HyperParameter object may be passed; it will be converted to a tuple of length 1.
            Args (Initialization Option 2):
                hp_dict (dict, optional): dictionary containing names of each HyperParameter as keys, mapping to a dictionary 
                    listing all attributes for that HyperParameter (any attributes not set will be assigned the default values in HyperParameter init).
        """

        if hp_set is not None:
            if hasattr(hp_set,'__iter__'):
                self.hyper_parameters = hp_set
            else:
                self.hyper_parameters = (hp_set,)
        else:
            if hp_dict is None:
                raise ValueError('Input must be provided to initialize HyperParameterSet!')
            self.__gen_hp_set(hp_dict)


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


    def set_values(self, ind):
        """Sets value of all HyperParameter objects in set to value at a specific index within the list of values.

            Args:
                ind (int): Index of object attribute values to use as new self.value.
        """

        for hp in self.hyper_parameters:
            hp.value = hp.values[ind]


    def to_dict(self):
        """Converts object data to a dict, where each key is the name of a HyperParameter, and each value is the HyperParameter's current value.

            Returns:
                dict: dictionary mapping HyperParameters in the set to their current values.
        """
        hp_dict = {hp.name:hp.value for hp in self.hyper_parameters}
        return hp_dict


    # PRIVATE METHODS

    def __gen_hp_set(self, hp_dict):
        # Converts a dict of dicts (where each key is the name of a HyperParameter, and each value is the attributes for that HyperParameter)
        # into a list of HyperParameter objects.
        hp_set = []

        for key, values in hp_dict.items():
            hyper_parameter = HyperParameter(name=key, **values)
            hp_set.append(hyper_parameter)

        self.hyper_parameters = tuple(hp_set)
