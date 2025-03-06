from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_net import HyperParameterSet


@dataclass
class ScenarioObjects:
    # CONSTRUCTOR
    save_options: dict = None  # Not currently used. TODO: should use this
    datasets: dict = None
    hyperparameters: HyperParameterSet = None
    predictors: dict = None
    tuners: dict = None
    tunings: dict = None  # Not currently used
    trainings: dict = None  # Not currently used
    evaluations: dict = None
    gamblers: dict = None  # Not currently used
    plot_groups: dict = None  # Not currently used

    def get_obj_by_name(self, name, category=None):
        # "Null result" = return the input as the output (couldn't find it)
        value = name

        for sub_cat, sub_class in vars(self).items():
            if (category is None or category == sub_cat) and (hasattr(sub_class, "__iter__") and name in sub_class):
                value = sub_class[name]
                break

        return value
