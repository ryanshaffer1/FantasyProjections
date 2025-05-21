from __future__ import annotations

import pandas as pd

from config.player_id_config import PRIMARY_PLAYER_ID
from data_pipeline.features.feature_set import FeatureSet
from data_pipeline.utils.data_helper_functions import subsample_game_time


class PlayerInfoFeatureSet(FeatureSet):
    def __init__(self, features, sources):
        super().__init__(features, sources)
        self.raw_rosters_df = None

    def collect_data(
        self,
        year: list[int] | range | int,
        weeks: list[int] | range,
        df_sources: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        super().collect_data(year, weeks, df_sources)
        self.raw_rosters_df = next(iter(self.df_dict.values()))

    def process_data(self, game_data_worker):
        # Compute stats for each player on the team in this game
        list_of_player_dfs = game_data_worker.roster_df.reset_index().apply(
            self.midgame_player_info,
            args=(game_data_worker,),
            axis=1,
        )
        stats_df = pd.concat(list_of_player_dfs.tolist())

        # Add player info to dataframe
        stats_df["Position"] = game_data_worker.roster_df.loc[(stats_df[PRIMARY_PLAYER_ID], "Position")].tolist()
        stats_df["Age"] = game_data_worker.roster_df.loc[(stats_df[PRIMARY_PLAYER_ID], "Age")].tolist()

        # Set common index
        stats_df[["Year", "Week"]] = [game_data_worker.year, game_data_worker.week]
        stats_df = stats_df.reset_index().set_index(["Year", "Week", PRIMARY_PLAYER_ID, "Elapsed Time"])

        return stats_df

    def midgame_player_info(self, player_info, game_data_worker):
        """Determines the mid-game info for one player throughout the game.

            Args:
                player_info (pandas.Series): Roster information for one player (number, ID, position, etc.).
                game_times (list | str): Times in the game to output midgame stats for. May be a list of elapsed times in minutes,
                    in which case the soonest play-by-play time after that elapsed time will be used as the stats at that time. If a string is passed, no filtering occurs.

            Returns:
                pandas.DataFrame: Player's info at each time in the game (including any time-varying data). May have an additional (redundant) row for the final game time.

        """  # fmt: skip

        # Set up dataframe covering player's contributions each play
        player_stats_df = pd.DataFrame()  # Output array
        # Game time elapsed
        player_stats_df["Elapsed Time"] = game_data_worker.pbp_df.reset_index()["Elapsed Time"]
        player_stats_df = player_stats_df.set_index("Elapsed Time")

        # Add some player info
        player_stats_df[PRIMARY_PLAYER_ID] = player_info[PRIMARY_PLAYER_ID]

        # Trim to just the game times of interest
        player_stats_df = subsample_game_time(player_stats_df, game_data_worker.game_times)

        return player_stats_df
