"""Base class used to collect/process/store groups of related data "features".

    Class:
        FeatureSet : Class storing configuration for how to collect and process data for multiple related features, such as player info.

"""  # fmt: skip

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from misc.manage_files import collect_input_dfs

if TYPE_CHECKING:
    from data_pipeline.features.feature import Feature


class FeatureSet:
    """Class storing configuration for how to collect and process data for multiple related features, such as player info.

        This is a base class providing common functions to child classes, which must implement unique logic to process data.

        Args:
            features (list[Feature]): All Feature (or sub-classes of Feature) objects to include in the feature set.
            sources (dict): Paths to both local and online sources to collect data associated with the feature set.

        Additional Class Attributes:
            thresholds (dict[str, list]): Maps each individual feature in the set to its normalization thresholds
            df_dict (dict): Stores loaded dataframes (including previously-cached dataframes) associated with each data source.

        Public Methods:
            post_init : Performs initialization actions after some more steps have been completed.
            collect_data : Performs collection of data related to the feature set by searching the sources and checking for completeness.
            process_data : Performs all the processing of collected info needed to produce the outputs related to this feature set.
            collect_validation_data : Gathers data needed for independent verification of the feature set's data.

    """  # fmt: skip

    def __init__(
        self,
        features: list[Feature],
        sources: dict[str, dict[str, str]] | dict[str, str],
    ) -> None:
        """Constructor for FeatureSet objects.

            Args:
                features (list[Feature]): All Feature (or sub-classes of Feature) objects to include in the feature set.
                sources (dict): Paths to both local and online sources to collect data associated with the feature set.

        """  # fmt: skip
        self.sources = sources
        self.features = features
        self.thresholds = {feat.name: feat.thresholds for feat in self.features}
        self.df_dict = {}

    def post_init(self, **kwargs):
        """Performs initialization actions after some more steps have been completed.

            1. Input YAML file has been fully parsed.
            2. Data files configuration has been loaded.
            3. Output arrays have been initialized.
            4. Roster filter has (maybe) been initialized.

            Not used by the base FeatureSet class, but may be used by child classes.

        """  # fmt: skip

    def collect_data(
        self,
        year: int,
        weeks: list[int] | range,
        df_sources: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Performs collection of data related to the feature set by searching the sources and checking for completeness.

            Args:
                year (int): Year associated with the data (assumes the full year's worth of data is contained in one file).
                weeks (list | range): Weeks to ensure are included in the collected data (will search for them online if not).
                df_sources (dict, optional): Cached map of filenames to dataframes that have already been loaded (reduces re-loading). Defaults to None.

        """  # fmt: skip

        # Optional input
        if df_sources is None:
            df_sources = {}
        # Collect data from local or online source

        if isinstance(self.sources["local"], dict):
            # Check if multiple sources are already specified in a dict
            local_file_path = self.sources["local"].copy()
        else:
            # If only one source, format it into a dict
            local_file_path = {"feature_set": self.sources["local"]}
        if isinstance(self.sources["online"], dict):
            # Check if multiple sources are already specified in a dict
            online_file_path = self.sources["online"].copy()
        else:
            # If only one source, format it into a dict
            online_file_path = {"feature_set": self.sources["online"]}

        # Look for a previously loaded dataframe from each desired input file
        for source, filename in local_file_path.items():
            if filename.format(year) in df_sources:
                self.df_dict[source] = df_sources[filename.format(year)]

        # Remove any sources that have already been loaded
        for source in self.df_dict:
            del local_file_path[source]
            del online_file_path[source]

        # Collect any remaining dataframes from the local/online file sources
        found_df_dict, found_df_sources = collect_input_dfs(year, weeks, local_file_path, online_file_path, online_avail=True)
        self.df_dict.update(found_df_dict[0])
        df_sources.update(found_df_sources)

    def process_data(self, *_args, **_kwargs):
        """Performs all the processing of collected info needed to produce the outputs related to this feature set.

            Not used by the base FeatureSet class, but MUST be defined by child classes.
        """  # fmt: skip

    def collect_validation_data(
        self,
        *_args,
        **_kwargs,
    ):
        """Gathers data needed for independent verification of the feature set's data.

            Not used by the base FeatureSet class, but may be used by child classes.

            Args:
                Any - defined by child class's function overload.

            Returns:
                pandas.DataFrame: Dataframe with all data required to perform validation.

        """  # fmt: skip

        return pd.DataFrame()
