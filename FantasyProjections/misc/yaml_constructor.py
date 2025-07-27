"""Functions to handle the loading/construction of custom classes from a YAML file.

    Functions:
        add_yaml_constructors : Add YAML constructors for custom tags used in the YAML configuration files.
        constructor : Custom constructor from YAML inputs for every class enumerated above.
        concat_constructor : Custom constructor for the !concat tag in YAML files, used to concatenate lists of strings.
        config_var_constructor : Custom constructor for the !config_var tag in YAML files to obtain variables defined in a separate dict.
        data_files_config_constructor : Custom constructor for the !DataFilesConfig tag in YAML files to load a DataFilesConfig object.
        path_constructor : Custom constructor for the !path tag in YAML files to concatenate filepaths.
        range_constructor : Custom constructor for the !range tag in YAML files to define a range from [start], stop, [step] inputs.

"""  # fmt: skip

from __future__ import annotations

import os

import yaml

from data_pipeline import features
from misc import yaml_dataclasses

# Following classes use generic "constructor" function (best suited for dataclasses)
constructable_classes = {
    "Flags": yaml_dataclasses.Flags,
    "DatasetOptions": yaml_dataclasses.DatasetOptions,
    "FeatureSet": features.FeatureSet,
    "GameContextFeatureSet": features.GameContextFeatureSet,
    "StatsFeatureSet": features.StatsFeatureSet,
    "PlayerInfoFeatureSet": features.PlayerInfoFeatureSet,
    "InjuryFeatureSet": features.InjuryFeatureSet,
    "OddsFeatureSet": features.OddsFeatureSet,
    "Feature": features.Feature,
    "StatFeature": features.StatFeature,
    "MultiColumnFeature": features.MultiColumnFeature,
    # Add other classes as needed
}


def add_yaml_constructors():
    """Add YAML constructors for custom tags used in the YAML configuration files.

    This allows for custom processing of specific tags when loading YAML files.
    """
    # Custom constructors for FeatureSet, Feature, and other basic classes
    for class_name in constructable_classes:
        yaml_tag = f"!{class_name}"
        yaml.add_constructor(yaml_tag, constructor, Loader=yaml.SafeLoader)

    # Custom constructors for YAML utilities
    yaml.add_constructor("!concat", concat_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!config_var", config_var_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!path", path_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!range", range_constructor, Loader=yaml.SafeLoader)

    # Custom constructors for classes that need their own implementation
    yaml.add_constructor("!DataFilesConfig", data_files_config_constructor, Loader=yaml.SafeLoader)


def constructor(loader, node):
    """Custom constructor from YAML inputs for every class enumerated above.

        Args:
            loader (yaml.SafeLoader): Loader used to generate object from YAML inputs.
            node (yaml.MappingNode): YAML input parameters describing the object.

        Returns:
            any: Object of any type that uses this constructor function (enumerated above).

    """  # fmt: skip

    class_type = constructable_classes[node.tag.replace("!", "")]
    value = loader.construct_mapping(node, deep=True)
    return class_type(**value)


def concat_constructor(loader, node):
    """Custom constructor for the !concat tag in YAML files, used to concatenate lists of strings.

        Args:
            loader (yaml.SafeLoader): Loader used to generate object from YAML inputs.
            node (yaml.SequenceNode): YAML input parameters listing the strings to concat.

        Returns:
            str: Concatenated string

    """  # fmt: skip
    sequence = loader.construct_sequence(node, deep=True)
    str_concat = "".join(sequence)
    return str_concat


def config_var_constructor(loader, node):
    """Custom constructor for the !config_var tag in YAML files to obtain variables defined in a separate dict.

        Primarily used for obtaining data from data_files_config, which is loaded into the YAML at runtime
        and thus cannot set anchors to individual keys.

        Args:
            loader (yaml.SafeLoader): Loader used to generate object from YAML inputs.
            node (yaml.SequenceNode): YAML input parameters:
                - Element 0: anchor to the YAML entry to pull from.
                - Element 1: name of the key to search for in the anchored entry.

        Returns:
            any: Value for the specified key in the specified dict.

    """  # fmt: skip

    [data_files_config, config_var_key] = loader.construct_sequence(node, deep=False)
    config_var = data_files_config[config_var_key]
    return config_var


def data_files_config_constructor(loader, node):
    """Custom constructor for the !DataFilesConfig tag in YAML files to load a DataFilesConfig object.

        Special handling required because the DataFilesConfig object is not returned,
        only a dict of its config variables.

        Args:
            loader (yaml.SafeLoader): Loader used to generate object from YAML inputs.
            node (yaml.MappingNode): YAML input parameters for the DataFilesConfig object (path to its own config file)

        Returns:
            dict: Key-value pairs read from the DataFilesConfig object's config file.

    """  # fmt: skip
    """Custom constructor for the !DataFilesConfig tag in YAML files.
    This allows for loading a DataFilesConfig object from a YAML file.
    """
    config_file = loader.construct_mapping(node)["config_file"]
    return yaml_dataclasses.DataFilesConfig(config_file=config_file).config


def path_constructor(loader, node):
    """Custom constructor for the !path tag in YAML files to concatenate filepaths.

        Args:
            loader (yaml.SafeLoader): Loader used to generate object from YAML inputs.
            node (yaml.SequenceNode): List of filepath elements to concatenate.

        Returns:
            str: Fully-concatenated file path (complying with OS filepath rules).

    """  # fmt: skip

    sequence = loader.construct_sequence(node, deep=True)
    path_concat = os.path.join(*sequence)
    return path_concat


def range_constructor(loader, node):
    """Custom constructor for the !range tag in YAML files to define a range from [start], stop, [step] inputs.

        Args:
            loader (yaml.SafeLoader): Loader used to generate object from YAML inputs.
            node (yaml.SequenceNode): List of inputs, interpreted based on number of inputs:
                - 1 input: node = [stop]
                - 2 inputs: node = [start, stop]
                - 3 inputs: node = [start, stop, step]

        Raises:
            ValueError: Input more than 3 elements or less than 1 element.

        Returns:
            list: list of numbers in the range.

    """  # fmt: skip

    sequence = loader.construct_sequence(node, deep=True)
    match len(sequence):
        case 1:
            start = 0
            stop = sequence[0]
            step = 1
        case 2:
            start = sequence[0]
            stop = sequence[1]
            step = 1
        case 3:
            start = sequence[0]
            stop = sequence[1]
            step = sequence[2]
        case _:
            msg = "Invalid range format. Expected [start], stop, [step]."
            raise ValueError(msg)

    list_range = list(range(start, stop, step))

    return list_range
