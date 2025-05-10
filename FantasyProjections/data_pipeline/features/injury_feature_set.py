from data_pipeline.features.feature_set import FeatureSet


class InjuryFeatureSet(FeatureSet):
    def __init__(self, name, sources, thresholds):
        super().__init__(name, sources, thresholds)
        self.df = None

    def collect_data(self, year, weeks):
        super().collect_data(year, weeks)
        self.df = self.df_dict[self.name]

    def process_weekly_updates(self, seasonal_stats_collector):
        """Adds injury data to the all_rosters_df attribute.

            Args:
                roster_df (pandas.DataFrame): Contains weekly roster for all NFL teams.

            Returns:
                pandas.DataFrame: Roster dataframe with injury data added.

        """  # fmt: skip

        # Handle no injury data collected
        if self.df is None:
            return None

        # Clean up injury data
        injury_df = self.df.drop_duplicates(subset=["gsis_id", "week"])

        # Quantified injury status
        injury_scale = {"Out": 0, "Doubtful": 0.25, "Questionable": 0.5, "Probable": 0.75, "Active": 1}

        # Collect a list of injury status by week/player in roster_df
        injury_status = seasonal_stats_collector.all_rosters_df.merge(
            injury_df,
            left_on=["gsis_id", "Week"],
            right_on=["gsis_id", "week"],
            how="left",
        )["report_status"]

        # Set common index and useful column name
        injury_status = injury_status.rename("injury_status")
        injury_status.index = seasonal_stats_collector.all_rosters_df.index

        # Map injury status to numerical value
        injury_status = injury_status.map(injury_scale).fillna(injury_scale["Active"])

        return injury_status

    def process_midgame_updates(self):
        pass
