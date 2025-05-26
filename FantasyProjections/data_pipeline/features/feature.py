from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Feature:
    # CONSTRUCTOR
    name: str
    thresholds: list
    outputs: list | None = None
    one_hot_encode: bool = False
    validate: bool = False

    def __post_init__(self):
        # Make sure midgame is in the outputs if one-hot encoding is True (this is the only output that gets one-hot encoded outputs)
        if self.one_hot_encode:
            if isinstance(self.outputs, list):
                if "midgame" not in self.outputs:
                    self.outputs.append("midgame")
            else:
                self.outputs = ["midgame"]


@dataclass
class StatFeature(Feature):
    # CONSTRUCTOR
    scoring_weight: float = 0.0
    validate: bool = True
    site_labels: dict | None = None
