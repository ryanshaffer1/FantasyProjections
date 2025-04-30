"""Creates and exports a class that stores and manages nearly all FantasyProjections scenario-related data and objects.

    Classes:
        ScenarioObjects : Class storing and manipulating all objects and processes involved in the FantasyProjections scenario.
"""  # fmt: skip

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_net import HyperParameterSet


@dataclass
class ScenarioObjects:
    """Class storing and manipulating all objects and processes involved in the FantasyProjections scenario.

        Args:
            save_options (dict): flags and settings for saving various scenario outputs (models, plots, etc.) to file.
            datasets (dict): maps dataset names to StatsDataset objects.
            hyperparameters (HyperParameterSet): names and configurations of all model hyper-parameter controlled by tuners.
            predictors (dict): maps predictor names to FantasyPredictor objects.
            tuners (dict): maps tuner names to HyperParamTuner objects.
            tunings (dict): not currently implemented.
            trainings (dict): not currently implemented.
            evaluations (dict): maps evaluation names to PredictionResult objects.
            gamblers (dict): not currently implemented.
            plot_groups (dict): not currently implemented.

        Public Methods:
            get_obj_by_name : Finds an object (e.g. a StatsDataset, FantasyPredictor, etc.) in the ScenarioObjects which matches an input name.

    """  # fmt: skip

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
        """Finds an object (e.g. a StatsDataset, FantasyPredictor, etc.) in the ScenarioObjects which matches an input name.

            Args:
                name (str): name of the desired object, as defined in the input file (e.g. "name" attribute in a "datasets" entry).
                category (str, optional): Name of category (e.g. "dataset") to search for the object.
                    Defaults to None - all categories will be searched.

            Returns:
                object: Scenario object (such as a StatsDataset) which matches the input name.

        """  # fmt: skip

        # "Null result" = return the input as the output (couldn't find it)
        value = name

        for sub_cat, sub_class in vars(self).items():
            if (category is None or category == sub_cat) and (hasattr(sub_class, "__iter__") and name in sub_class):
                value = sub_class[name]
                break

        return value
