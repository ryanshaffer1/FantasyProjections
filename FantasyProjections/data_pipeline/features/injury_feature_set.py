from __future__ import annotations

from typing import TYPE_CHECKING

from data_pipeline.features.feature_set import FeatureSet

if TYPE_CHECKING:
    import pandas as pd


class InjuryFeatureSet(FeatureSet):
    def __init__(self, features, sources):
        super().__init__(features, sources)
        self.df = None

    def collect_data(
        self,
        year: list[int] | range | int,
        weeks: list[int] | range,
        df_sources: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        super().collect_data(year, weeks, df_sources)
        self.df = next(iter(self.df_dict.values()))

        # Clean up injury data
        self.df = self.df.drop_duplicates(subset=["gsis_id", "week"], keep="last")

    def process_data(self, game_data_worker):
        """Adds injury data to the all_rosters_df attribute.

            Args:
                roster_df (pandas.DataFrame): Contains weekly roster for all NFL teams.

            Returns:
                pandas.DataFrame: Roster dataframe with injury data added.

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
