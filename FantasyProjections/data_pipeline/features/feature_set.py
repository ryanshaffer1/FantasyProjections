from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from misc.manage_files import collect_input_dfs

if TYPE_CHECKING:
    from data_pipeline.features.feature import Feature


class FeatureSet:
    def __init__(self, features: list[Feature], sources: dict[str, dict[str, str]] | dict[str, str]) -> None:
        self.sources = sources
        self.features = features
        self.thresholds = {feat.name: feat.thresholds for feat in self.features}
        self.df_dict = {}

    def collect_data(
        self,
        year: list[int] | range | int,
        weeks: list[int] | range,
        df_sources: dict[str, pd.DataFrame] | None = None,
    ) -> None:
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

    def get_features_by_name(self, name: str) -> Feature:
        """Returns the feature object with the given name."""
        for feature in self.features:
            if feature.name == name:
                return feature
        msg = f"Feature {name} not found in feature set."
        raise ValueError(msg)

    def collect_validation_data(
        self,
        *_args,
        **_kwargs,
    ):
        return pd.DataFrame()
