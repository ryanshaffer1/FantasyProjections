"""Creates and exports a class that supports the handling and manipulation of Neural Network HyperParameters.

    Hyper-Parameter = Variable within ML equations which is not learned by the model during training, and must be set before training.

    Classes:
        HyperParameter : Class handling the value and variations of a Neural Net Hyper-Parameter.
"""  # fmt: skip

from dataclasses import dataclass

import numpy as np


@dataclass
class HyperParameter:
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
            randomize_in_range : Returns n random values (from a uniform distribution) within the HyperParameter's val_range.
            adjust_range : Returns new bounds for HyperParameter values that are centered on a given point and scaled up/down from val_range.

    """  # fmt: skip

    # CONSTRUCTOR
    name: str
    optimizable: bool = False
    value: int = None
    val_range: list = None
    val_scale: str = "none"
    values: list = None

    def __post_init__(self):
        # Evaluates as part of the Constructor.
        # Generates attributes that are not simple data copies of inputs.

        if self.value is None:
            self.value = 0

        if self.values is None:
            self.values = [self.value]  # This may be overwritten later, if optimizing hyper-parameters

        if not self.val_range or not self.optimizable:
            self.val_range = [self.value]
            self.val_scale = "none"

    # PUBLIC METHODS

    def copy(self):
        """Returns a copy of the HyperParameter object."""  # fmt: skip
        new_hp = HyperParameter(
            name=self.name,
            optimizable=self.optimizable,
            value=self.value,
            val_range=self.val_range,
            val_scale=self.val_scale,
            values=self.values,
        )
        return new_hp

    def randomize_in_range(self, n_values):
        """Returns n random values (from a uniform distribution) within the HyperParameter's val_range.

            Args:
                n_values (int): Number of random values to generate.

            Returns:
                list: List of random values.

        """  # fmt: skip

        values = [self.value] * n_values

        if isinstance(self.val_range, list):
            match self.val_scale:
                case "linear":
                    values = np.random.uniform(self.val_range[0], self.val_range[1], n_values).tolist()
                case "log":
                    uniform_vals = np.random.uniform(np.log10(self.val_range[0]), np.log10(self.val_range[1]), n_values)
                    values = (10**uniform_vals).tolist()
                case "none":
                    values = [self.value] * n_values
                case "selection":
                    values = np.random.choice(self.val_range, n_values)
                case _:
                    values = [self.value] * n_values

        return values

    def adjust_range(self, center_point, scale_factor=1, exceed_boundary=False):
        """Returns new bounds for HyperParameter values that are centered on a given point and scaled up/down from val_range.

            If self.val_scale is linear or log, this method changes the range per the appropriate scale.
            Otherwise, this method returns the input center point as a list.

            Args:
                center_point (int | float | other): Value to center new range on.
                scale_factor (int, optional): Multiplier to adjust size of the val_range. Defaults to 1.
                exceed_boundary (bool, optional): Whether to allow the new range to exceed the boundaries of the old range.
                    Defaults to False.

            Returns:
                list: Range of values, centered on the center point, scaled per scale_factor, and following the HP's val scale.

        """  # fmt: skip

        new_val_range = None
        match self.val_scale:
            case "linear":
                # Re-scale
                og_range_size = self.val_range[1] - self.val_range[0]
                new_range_size = og_range_size * scale_factor
                # Check if center point is on the edge of the range and boundaries cannot be exceeded
                if not exceed_boundary and (
                    (center_point - new_range_size / 2) < self.val_range[0]
                    or (center_point + new_range_size / 2) > self.val_range[1]
                ):
                    # Range is clipped to not exceed the limits in val_range
                    side_of_range = float(np.sign(center_point - np.mean(self.val_range)))
                    range_edge_to_keep = self.val_range[int((side_of_range + 1) / 2)]
                    new_point = range_edge_to_keep - (new_range_size * side_of_range)
                    new_val_range = sorted([range_edge_to_keep, new_point])
                else:
                    # Range is re-centered and scaled
                    new_val_range = [center_point - new_range_size / 2, center_point + new_range_size / 2]

            case "log":
                log_center_point = np.log10(center_point)
                log_val_range = np.log10(self.val_range)
                # Re-scale
                og_log_size = log_val_range[1] - log_val_range[0]
                new_log_size = og_log_size * scale_factor
                # Check if center point is on the edge of the range and boundaries cannot be exceeded
                if not exceed_boundary and (
                    (log_center_point - new_log_size / 2) < log_val_range[0]
                    or (log_center_point + new_log_size / 2) > log_val_range[1]
                ):
                    # Range is clipped to not exceed the limits in val_range
                    side_of_range = np.sign(log_center_point - np.mean(log_val_range))
                    new_point = 10 ** (log_center_point - (new_log_size * side_of_range))
                    new_val_range = sorted([float(center_point), float(new_point)])

                else:
                    # Range is re-centered and scaled
                    new_val_range = [
                        float(10 ** (log_center_point - new_log_size / 2)),
                        float(10 ** (log_center_point + new_log_size / 2)),
                    ]

            case _:
                # No setting a range, just return the center point
                new_val_range = [center_point]

        return new_val_range
