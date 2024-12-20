"""Creates and exports a class that supports the handling and manipulation of Neural Network HyperParameters.
    Hyper-Parameter = Variable within ML equations which is not learned by the model during training, and must be set before training.

    Classes:
        HyperParameter : Class handling the value and variations of a Neural Net Hyper-Parameter.
"""

from dataclasses import dataclass
from config.hp_config import hp_defaults

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
        
        Additional Class Attributes:
            values (list): Sequence of values to use as the value attribute over successive tuning iterations. Determined by a HyperParameterTuner.
                Unique values may repeat.

        Public Methods:
            copy : Returns a copy of the HyperParameterSet object.
    """

    # CONSTRUCTOR
    name: str
    optimizable: bool = False
    value: int = None
    val_range: list = None
    val_scale: str = 'none'
    values: list = None

    def __post_init__(self):
        # Evaluates as part of the Constructor.
        # Generates attributes that are not simple data copies of inputs.

        if self.value is None:
            self.value = hp_defaults.get(self.name, {}).get('value',0)

        if self.values is None:
            self.values = [self.value] # This may be overwritten later, if optimizing hyper-parameters

        if not self.val_range or not self.optimizable:
            self.val_range = [self.value]


    # PUBLIC METHODS

    def copy(self):
        """Returns a copy of the HyperParameter object.
        """
        new_hp = HyperParameter(name=self.name,
                                optimizable=self.optimizable,
                                value=self.value,
                                val_range=self.val_range,
                                val_scale=self.val_scale,
                                values=self.values)
        return new_hp
