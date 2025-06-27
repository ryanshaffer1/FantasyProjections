from __future__ import annotations

import os

import yaml

from config import data_files_config
from data_pipeline import features
from misc import yaml_dataclasses

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
    """
    Add YAML constructors for custom tags used in the YAML configuration files.
    This allows for custom processing of specific tags when loading YAML files.
    """
    # Custom constructors for FeatureSet and Feature objects
    for class_name in constructable_classes:
        yaml_tag = f"!{class_name}"
        yaml.add_constructor(yaml_tag, constructor, Loader=yaml.SafeLoader)

    # Custom constructors for YAML utilities
    yaml.add_constructor("!concat", concat_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!config_var", config_var_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!path", path_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!range", range_constructor, Loader=yaml.SafeLoader)


def constructor(loader, node):
    """
    Custom constructor for every class in this package that is constructable from YAML.
    """
    class_type = constructable_classes[node.tag.replace("!", "")]
    value = loader.construct_mapping(node, deep=True)
    return class_type(**value)


def concat_constructor(loader, node):
    """
    Custom constructor for the !concat tag in YAML files.
    This allows for concatenation of lists or strings.
    """
    sequence = loader.construct_sequence(node, deep=True)
    str_concat = "".join(sequence)
    return str_concat


def config_var_constructor(_loader, node):
    """
    Custom constructor for the !config_var tag in YAML files.
    This allows for using variables defined in another file.
    """
    config_var = getattr(data_files_config, node.value)
    return config_var


def path_constructor(loader, node):
    """
    Custom constructor for the !path tag in YAML files.
    This allows for concatenation of filepaths.
    """
    sequence = loader.construct_sequence(node, deep=True)
    path_concat = os.path.join(*sequence)
    return path_concat


def range_constructor(loader, node):
    """
    Custom constructor for the !range tag in YAML files.
    This allows for defining a range of numbers with inputs [start], stop, [step].
    """
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
