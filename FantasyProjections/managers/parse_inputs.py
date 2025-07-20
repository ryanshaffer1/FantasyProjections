"""Functions and classes used to manipulate user inputs into the FantasyProjections scenario.

    Functions:
        parse_inputs : Reads input YAML file into InputParameters object, normalizes, and optionally saves.
        recursive_dict_merge : Fills in any missing values in input_struct based on the values in defaults.

    Classes:
        InputParameters : Data structure used to store and manipulate all inputs passed into the scenario.

"""  # fmt:skip

import logging
import os

import yaml

from misc.dataset import StatsDataset
from misc.manage_files import create_folders, name_save_folder
from misc.yaml_constructor import add_yaml_constructors

default_filename = "FantasyProjections/config/default_inputs.yaml"


# Set up logger
logger = logging.getLogger("log")


def parse_inputs(input_filename):
    """Reads input YAML file into InputParameters object, normalizes, and optionally saves.

        Normalization: replacing any missing, required inputs with inputs from the default input file.

        Args:
            input_filename (str): YAML input file for current scenario

        Returns:
            InputParameters: Parsed and normalized inputs.

    """  # fmt: skip
    add_yaml_constructors()

    with open(input_filename) as stream:
        inputs = InputParameters(yaml.safe_load(stream))

    with open(default_filename) as stream:
        default_inputs = InputParameters(yaml.safe_load(stream))

    inputs.normalize(default_inputs)

    # Generate save directory
    inputs.save_options["save_directory"] = name_save_folder(inputs.save_options)

    # Save a copy of the input parameters if specified
    if inputs.save_options["save_input_file"]:
        inputs.save()

    return inputs


class InputParameters:
    """Data structure used to store and manipulate all inputs passed into the scenario.

        Class Attributes:
            save_options (dict): flags and settings for saving various scenario outputs (models, plots, etc.) to file.
            datasets (list[dict]): inputs for all StatsDataset objects to generate, including names and configurations.
            hyperparameters (dict): names and configurations of all model hyper-parameter controlled by tuners.
            predictors (list[dict]): inputs for all FantasyPredictor objects to generate, including names, types, and configurations.
            tuners (list[dict]): inputs for all HyperParamTuner objects to generate, including names, types, and configurations.
            tunings (list[dict]): inputs for all hyper-parameter tuning processes to perform, including tuners/predictors involved.
            trainings (list[dict]): inputs for all predictor training processes to perform, including predictors/datasets involved.
            evaluations (list[dict]): inputs for all predictor evaluation processes to perform, including predictors/datasets involved.
            gamblers (list[dict]): inputs for all Gambler objects to generate, including name, type, and evaluation to use.
            plot_groups (list[dict]): inputs for all plot groups to generate, including results to show and plot settings.

        Public Methods:
            normalize : Replace any missing, required inputs in the InputParameters with matching inputs from the default InputParameters.
            save : Generates YAML file from the InputParameters object.


    """  # fmt: skip

    def __init__(self, input_dict):
        """Constructor for InputParameters.

            Args:
                input_dict (dict): Dictionary parsed from an input YAML file. May contain the following fields:
                    save_options (dict): flags and settings for saving various scenario outputs (models, plots, etc.) to file.
                    datasets (list[dict]): inputs for all StatsDataset objects to generate, including names and configurations.
                    hyperparameters (dict): names and configurations of all model hyper-parameter controlled by tuners.
                    predictors (list[dict]): inputs for all FantasyPredictor objects to generate, including names, types, and configurations.
                    tuners (list[dict]): inputs for all HyperParamTuner objects to generate, including names, types, and configurations.
                    tunings (list[dict]): inputs for all hyper-parameter tuning processes to perform, including tuners/predictors involved.
                    trainings (list[dict]): inputs for all predictor training processes to perform, including predictors/datasets involved.
                    evaluations (list[dict]): inputs for all predictor evaluation processes to perform, including predictors/datasets involved.
                    gamblers (list[dict]): inputs for all Gambler objects to generate, including name, type, and evaluation to use.
                    plot_groups (list[dict]): inputs for all plot groups to generate, including results to show and plot settings.

            Class Attributes:
                All fields in the input_dict are converted to attributes of the InputParameters object.
                Any omitted fields are initialized as an empty dict or empty list, depending on the input type.

        """  # fmt: skip
        self.data_files_config = input_dict.get("data_files_config", {})
        self.save_options = input_dict.get("save_options", {})
        self.features = input_dict.get("features", {})
        self.datasets = input_dict.get("datasets", [])
        self.hyperparameters = input_dict.get("hyperparameters", {})
        self.predictors = input_dict.get("predictors", [])
        self.tuners = input_dict.get("tuners", [])
        self.tunings = input_dict.get("tunings", [])
        self.trainings = input_dict.get("trainings", [])
        self.evaluations = input_dict.get("evaluations", [])
        self.gamblers = input_dict.get("gamblers", [])
        self.plot_groups = input_dict.get("plot_groups", [])

    def normalize(self, default_inputs):
        """Replace any missing, required inputs in the InputParameters with matching inputs from the default InputParameters.

            Args:
                default_inputs (InputParameters): data structure parsed from the default input YAML file.

        """  # fmt: skip

        # Merge dictionaries, with self taking precedence over default_inputs in the case of matching keys
        # Note that though the recursive_dict_merge returns a value, it does not need to be assigned due to the memory persistence of dicts
        recursive_dict_merge(self.data_files_config, default_inputs.data_files_config, add_if_empty=True)
        recursive_dict_merge(self.save_options, default_inputs.save_options, add_if_empty=True)
        recursive_dict_merge(self.features, default_inputs.features, add_if_empty=True)
        recursive_dict_merge(self.datasets, default_inputs.datasets, add_if_empty=True)
        recursive_dict_merge(self.hyperparameters, default_inputs.hyperparameters, add_if_empty=True)
        recursive_dict_merge(self.predictors, default_inputs.predictors, add_if_empty=True)
        recursive_dict_merge(self.tuners, default_inputs.tuners, add_if_empty=False)
        recursive_dict_merge(self.tunings, default_inputs.tunings, add_if_empty=False)
        recursive_dict_merge(self.trainings, default_inputs.trainings, add_if_empty=False)
        recursive_dict_merge(self.evaluations, default_inputs.evaluations, add_if_empty=False)
        recursive_dict_merge(self.gamblers, default_inputs.gamblers, add_if_empty=False)
        recursive_dict_merge(self.plot_groups, default_inputs.plot_groups, add_if_empty=False)

    def save(self, save_file=None):
        """Generates YAML file from the InputParameters object.

            Args:
                save_file (str, optional): Filename (including path) to save YAML.
                    Defaults to filename "inputs_normalized.yaml" in the folder identified by the "save_directory" input.

        """  # fmt: skip

        # If no save file provided, generate default
        if save_file is None:
            save_file = os.path.join(self.save_options["save_directory"], "inputs_normalized.yaml")

        # Create folder if it does not exist
        create_folders(save_file)

        # Write InputParameters object to a YAML
        with open(save_file, "w") as file:
            yaml.dump(self, file)

    def update_params_based_on_features(self, all_data: StatsDataset) -> None:
        # Update NeuralNetwork shape based on input/output features
        for pred in self.predictors:
            if pred.get("type") == "NeuralNetPredictor":
                nn_shape = pred["config"]["nn_shape"]

                # Get number of inputs to each embedding layer
                embedding_inputs = {}
                embedding_indices = {}
                for e_name, e_layer in nn_shape["embedding"].items():
                    col_indices = [i for i, col in enumerate(all_data.x_data_columns) if col.startswith(f"{e_layer['feature']}_")]
                    embedding_inputs[e_name] = len(col_indices)
                    embedding_indices[e_name] = col_indices

                # Build inputs
                nn_shape["input"] = {}
                nn_shape["input"]["unembedded"] = len(all_data.x_data_columns) - sum(embedding_inputs.values())
                nn_shape["input"].update(embedding_inputs)

                # Track input indices in the nn_shape
                all_embedded_indices = {x for ind_list in embedding_indices.values() for x in ind_list}
                nn_shape["input_indices"] = {}
                nn_shape["input_indices"]["unembedded"] = [
                    x for x in range(len(all_data.x_data_columns)) if x not in all_embedded_indices
                ]
                nn_shape["input_indices"].update(embedding_indices)

                # Reformat embedding layers to just the sizes
                nn_shape["embedding"] = {e_name: e_layer["n"] for e_name, e_layer in nn_shape["embedding"].items()}

                # Set output size
                nn_shape["output"] = len(all_data.y_data_columns)


# ruff: noqa: PLR0912
def recursive_dict_merge(input_struct, defaults, add_if_empty=True):
    """Fills in any missing values in input_struct based on the values in defaults.

        Calls recursively so that nested dicts/lists/tuples are also filled with default values.
        Note that list/tuple inputs are only used to search for nested dicts - all the "base data" is assumed to be in dicts.

        Args:
            input_struct (dict | list | tuple): Structure with potentially missing data fields
            defaults (dict | list | tuple): Structure with the "master copy" of all necessary data fields
                and their default values.
            add_if_empty (bool, optional): If input_struct is empty, toggle whether it should take all values from defaults.
                Defaults to True.

        Returns:
            dict | list | tuple: input_struct, with any missing fields filled in with the corresponding values in defaults.

    """  # fmt: skip
    # Do nothing if the input or the default is not a searchable/mergeable data type
    mergeable_types = dict | list | tuple
    if not (isinstance(input_struct, mergeable_types) and isinstance(defaults, mergeable_types)):
        return input_struct

    # Do nothing if the input_struct is empty and add_if_empty is False
    if len(input_struct) == 0 and not add_if_empty:
        return input_struct

    # If input is a list, run the dict merge for each entry in the list
    if isinstance(input_struct, list | tuple):
        for input_entry in input_struct:
            recursive_dict_merge(input_entry, defaults)
        return input_struct

    # If "type" is a key for the input, then must match to the correct typed default
    if "type" in input_struct:
        defaults = _recursive_find_dict_of_matching_type(input_struct, defaults)
        if defaults is None:
            # Default dict with the same type could not be found - cannot merge
            logger.warning(f"No default inputs found for input of type {input_struct['type']}")
            return input_struct

    # If defaults is a list, run the dict merge against each default entry
    if isinstance(defaults, list | tuple):
        for default_entry in defaults:
            recursive_dict_merge(input_struct, default_entry)
        return input_struct

    # PAYLOAD: Iterate over all the default value keys
    for key in defaults:
        # If the key is present in the input, check if it is a dict, list, or tuple (and should be recursively merged.)
        # Otherwise, do nothing
        if key in input_struct:
            if isinstance(input_struct[key], dict) and isinstance(defaults[key], dict):
                recursive_dict_merge(input_struct[key], defaults[key])
            elif isinstance(input_struct[key], list | tuple) and isinstance(defaults[key], list | tuple):
                for item in input_struct[key]:
                    recursive_dict_merge(item, defaults[key])
        # If the key is not present in the input, add the default key/value to the input
        else:
            input_struct[key] = defaults[key]

    return input_struct


def _recursive_find_dict_of_matching_type(input_dict, defaults):
    # If input_dict has key "type", then it likely has type-specific inputs (and default values).
    # Search for a data structure in defaults (may be nested) that has the same type as input_dict.

    # If defaults is a dict, assume this is the level "type" should be at (no nested dicts in dicts... why?)
    if isinstance(defaults, dict):
        if input_dict["type"] == defaults.get("type"):
            return defaults
        return None

    # If defaults is a list or tuple, search each element for a matching type via recursion
    if isinstance(defaults, list | tuple):
        for item in defaults:
            default_found = _recursive_find_dict_of_matching_type(input_dict, item)
            if default_found is not None:
                return default_found

    return None
