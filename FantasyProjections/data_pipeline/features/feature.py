from dataclasses import dataclass


@dataclass
class Feature:
    # CONSTRUCTOR
    name: str
    thresholds: list
    outputs: list = None
    one_hot_encode: bool = False
    validate: bool = False


@dataclass
class StatFeature(Feature):
    # CONSTRUCTOR
    scoring_weight: float = 0.0
    validate: bool = True
    site_labels: dict = None
