"""Class used to collect/process/store data related to player injuries.

    Class:
        InjuryFeatureSet : Class that collects and processes data related to player injuries.

"""  # fmt: skip

from __future__ import annotations

from typing import TYPE_CHECKING

from data_pipeline.features.feature_set import FeatureSet

if TYPE_CHECKING:
    import pandas as pd


class InjuryFeatureSet(FeatureSet):
    """Class that collects and processes data related to player injuries.

        Sub-class of FeatureSet.

        Args:
            features (list[Feature]): All Feature (or sub-classes of Feature) objects to include in the feature set.
            sources (dict): Paths to both local and online sources to collect data associated with the feature set.

        Additional Class Attributes:
            thresholds (dict[str, list]): Maps each individual feature in the set to its normalization thresholds
            df_dict (dict): Stores loaded dataframes (including previously-cached dataframes) associated with each data source.
            df (pandas.DataFrame): Injury data, as collected from data sources.

        Public Methods:
            collect_data : See FeatureSet
            process_data : Generates injury status output data for each player based on the collected injury information.

    """  # fmt: skip

    def __init__(self, features, sources):
        """Constructor for InjuryFeatureSet objects.

            Args:
                features (list[Feature]): All Feature (or sub-classes of Feature) objects to include in the feature set.
                sources (dict): Paths to both local and online sources to collect data associated with the feature set.

        """  # fmt: skip

        super().__init__(features, sources)
        self.df = None

    def collect_data(
        self,
        year: int,
        weeks: list[int] | range,
        df_sources: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Collects data related to injury status by searching the provided sources and checking for completeness.

            Modifies the attribute "df" to store the collected injury data.

            Args:
                year (int): Year associated with the data (assumes the full year's worth of data is contained in one file).
                weeks (list | range): Weeks to ensure are included in the collected data (will search for them online if not).
                df_sources (dict, optional): Cached map of filenames to dataframes that have already been loaded (reduces re-loading). Defaults to None.

        """  # fmt: skip

        super().collect_data(year, weeks, df_sources)
        self.df = next(iter(self.df_dict.values()))

        # Clean up injury data
        self.df = self.df.drop_duplicates(subset=["gsis_id", "week"], keep="last")

    def process_data(self, game_data_worker):
        """Generates injury status output data for each player based on the collected injury information.

            Args:
                game_data_worker (SingleGameDataWorker): Processor for the current game, containing info on the roster, etc.

            Returns:
                pandas.DataFrame: Injury status for each player throughout this game. Indexed on Year, Week, Player ID, and Elapsed Time.

        """  # fmt: skip

        # Handle no injury data collected
        if self.df is None:
            return None

        # Quantified injury status
        injury_scale = {"Out": 0, "Doubtful": 0.25, "Questionable": 0.5, "Probable": 0.75, "Active": 1}

        # Collect a list of injury status by week/player in roster_df
        injury_status = game_data_worker.midgame_df.merge(
            self.df,
            left_on=["gsis_id", "Week"],
            right_on=["gsis_id", "week"],
            how="left",
        )["report_status"]

        # Set common index and useful column name
        injury_status = injury_status.rename(self.features[0].name)
        injury_status.index = game_data_worker.midgame_df.index

        # Map injury status to numerical value
        injury_status = injury_status.map(injury_scale).fillna(injury_scale["Active"])

        return injury_status
