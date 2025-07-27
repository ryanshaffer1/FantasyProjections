"""Classes that store configuration for how to collect and process data related to individual "features" such as stats, player information, etc.

    Classes:
        Feature : Class storing configuration for how to collect and process data regarding an individual feature, such as Pass Yds.
        StatsFeature : Class storing configuration for how to collect and process stats-related data (for one individual stat).
        MultiColumnFeature : Class storing configuration for how to collect and process data regarding an individual feature which is output in multiple columns.

"""  # fmt: skip

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Feature:
    """Class storing configuration for how to collect and process data regarding an individual feature, such as Pass Yds.

        All Features must be included in a FeatureSet which provides the unique logic required to process the data.

        Args:
            name (str): name of the feature, used to save as a column in output dataframes.
            thresholds (list): minimum and maximum values the feature can take on, used for data normalization.
            outputs (list, optional): Dataframes to include the feature in (Options are "id", "midgame", and "final"). Defaults to None.
            one_hot_encode (bool, optional): Whether to store the feature in Neural Net-readable dataframes as one-hot-encoded. Defaults to False.
                One-hot encoding creates a set of columns, each corresponding to the values a feature can take on. In each row, the column associated
                with the correct value is given a 1, and all other columns are given a 0.
            validate (bool, optional): Whether to perform extra independent verification of the collected data. Defaults to False.

        Additional Class Attributes:
            columns (list[str]): names of all columns in the output dataframes associated with this feature.

    """  # fmt: skip

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

        # List of columns associated with this feature in the output dataframe
        self.columns = [self.name]


@dataclass
class StatFeature(Feature):
    """Class storing configuration for how to collect and process stats-related data (for one individual stat).

        Sub-class of Feature.

        Args:
            name (str): name of the feature, used to save as a column in output dataframes.
            thresholds (list): minimum and maximum values the feature can take on, used for data normalization.
            scoring_weight (float, optional): weight value to use in computing Fantasy scores based on the feature. Defaults to 0.
            outputs (list, optional): Dataframes to include the feature in (Options are "id", "midgame", and "final"). Defaults to None.
            one_hot_encode (bool, optional): Whether to store the feature in Neural Net-readable dataframes as one-hot-encoded. Defaults to False.
                One-hot encoding creates a set of columns, each corresponding to the values a feature can take on. In each row, the column associated
                with the correct value is given a 1, and all other columns are given a 0.
            validate (bool, optional): Whether to perform extra independent verification of the collected data. Defaults to True.
            site_labels (dict, optional): Map between data sources and the name of this stat within those sources (to handle identical data with unique labels).

        Additional Class Attributes:
            columns (list[str]): names of all columns in the output dataframes associated with this feature.

    """  # fmt: skip

    # CONSTRUCTOR
    scoring_weight: float = 0.0
    validate: bool = True
    site_labels: dict | None = None


@dataclass
class MultiColumnFeature(Feature):
    """Class storing configuration for how to collect and process data regarding an individual feature which is output in multiple columns.

        Sub-class of Feature.

        Args:
            name (str): name of the feature, used to save as a column in output dataframes.
            thresholds (list): minimum and maximum values the feature can take on, used for data normalization.
            outputs (list, optional): Dataframes to include the feature in (Options are "id", "midgame", and "final"). Defaults to None.
            one_hot_encode (bool, optional): Whether to store the feature in Neural Net-readable dataframes as one-hot-encoded. Defaults to False.
                One-hot encoding creates a set of columns, each corresponding to the values a feature can take on. In each row, the column associated
                with the correct value is given a 1, and all other columns are given a 0.
            validate (bool, optional): Whether to perform extra independent verification of the collected data. Defaults to False.
            sub_columns (list[str], optional): Names of all columns to output that are associated with the feature. Defaults to None.
                Output columns are named as "name sub_column_name", with a space between the overall feature name and the sub-column.

        Additional Class Attributes:
            columns (list[str]): names of all columns in the output dataframes associated with this feature.

    """  # fmt: skip

    # CONSTRUCTOR
    sub_columns: list[str] | None = None

    def __post_init__(self):
        if self.sub_columns is None:
            self.columns = [self.name]
        else:
            self.columns = [f"{self.name} {sub_col}" for sub_col in self.sub_columns]
