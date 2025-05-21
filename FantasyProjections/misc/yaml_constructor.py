import os

import yaml

from config import data_files_config
from data_pipeline import features

constructable_classes = {
    "FeatureSet": features.FeatureSet,
    "BasicFeatureSet": features.BasicFeatureSet,
    "InjuryFeatureSet": features.InjuryFeatureSet,
    "Feature": features.Feature,
    "StatFeature": features.StatFeature,
    # Add other classes as needed
}


def add_yaml_constructors():
    """
    Add YAML constructors for custom tags used in the YAML configuration files.
    This allows for custom processing of specific tags when loading YAML files.
    """
    # Custom constructors for FeatureSet and Feature objects
    yaml.add_constructor("!Feature", constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!StatFeature", constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!BasicFeatureSet", constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!InjuryFeatureSet", constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!FeatureSet", constructor, Loader=yaml.SafeLoader)

    # Custom constructors for YAML utilities
    yaml.add_constructor("!concat", concat_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!path", path_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!config_var", config_var_constructor, Loader=yaml.SafeLoader)


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


def path_constructor(loader, node):
    """
    Custom constructor for the !path tag in YAML files.
    This allows for concatenation of filepaths.
    """
    sequence = loader.construct_sequence(node, deep=True)
    path_concat = os.path.join(*sequence)
    return path_concat


def config_var_constructor(_loader, node):
    """
    Custom constructor for the !config_var tag in YAML files.
    This allows for using variables defined in another file.
    """
    config_var = getattr(data_files_config, node.value)
    return config_var
