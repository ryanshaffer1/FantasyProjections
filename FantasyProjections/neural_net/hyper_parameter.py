"""Creates and exports a class that supports the handling and manipulation of Neural Network HyperParameters.
    Hyper-Parameter = Variable within ML equations which is not learned by the model during training, and must be set before training.

    Classes:
        HyperParameter : Class handling the value and variations of a Neural Net Hyper-Parameter.
"""

from dataclasses import dataclass, field, InitVar
import numpy as np
from config.hp_config import hp_tuner_settings, hp_defaults

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
            optimizable (bool, optional): Whether the hyper-parameter can be modified by a HyperParameter Tuner. (Whether it is actually optimized depends on the tuner.)
                Note that optimizable is not tracked as an object attribute, being redundant to other data. Defaults to False.
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
    optimizable: InitVar[bool] = False
    value: int = None
    values: list = field(init=False)
    val_range: list = None
    val_scale: str = 'none'
    num_steps: int = hp_tuner_settings['hyper_tuner_steps_per_dim']

    def __post_init__(self, optimizable):
        # Evaluates as part of the Constructor.
        # Generates attributes that are not simple data copies of inputs.

        if self.value is None:
            self.value = hp_defaults.get(self.name, {}).get('value',0)

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
                                                0, self.num_steps - 1], self.val_range).tolist()
                case 'log':
                    self.gridpoints = (10**np.interp(
                        range(self.num_steps), [0, self.num_steps - 1], np.log10(self.val_range))).tolist()
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
