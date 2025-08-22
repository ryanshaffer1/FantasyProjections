"""Class used to collect/process/store data related to player info, such as age, position, etc.

    Class:
        PlayerInfoFeatureSet : Class that collects and processes data related to player info, including age, position, etc.

"""  # fmt: skip

from __future__ import annotations

import pandas as pd

from config.player_id_config import PRIMARY_PLAYER_ID
from data_pipeline.features.feature_set import FeatureSet
from misc.data_helper_functions import subsample_game_time


class PlayerInfoFeatureSet(FeatureSet):
    """Class that collects and processes data related to player info, including age, position, etc.

        Sub-class of FeatureSet.

        Args:
            features (list[Feature]): All Feature (or sub-classes of Feature) objects to include in the feature set.
            sources (dict): Paths to both local and online sources to collect data associated with the feature set.

        Additional Class Attributes:
            thresholds (dict[str, list]): Maps each individual feature in the set to its normalization thresholds
            df_dict (dict): Stores loaded dataframes (including previously-cached dataframes) associated with each data source.

        Public Methods:
            collect_data : See FeatureSet
            process_data : Generates game context output data, such as team record and score, for each player based on the collected play-by-play data.

    """  # fmt: skip

    def __init__(self, features, sources):
        """Constructor for PlayerInfoFeatureSet objects.

            Args:
                features (list[Feature]): All Feature (or sub-classes of Feature) objects to include in the feature set.
                sources (dict): Paths to both local and online sources to collect data associated with the feature set.

        """  # fmt: skip

        super().__init__(features, sources)

    def collect_data(
        self,
        year: int,
        weeks: list[int] | range,
        df_sources: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Collects data related to game context by searching the provided sources and checking for completeness.

            Args:
                year (int): Year associated with the data (assumes the full year's worth of data is contained in one file).
                weeks (list | range): Weeks to ensure are included in the collected data (will search for them online if not).
                df_sources (dict, optional): Cached map of filenames to dataframes that have already been loaded (reduces re-loading). Defaults to None.

        """  # fmt: skip

        super().collect_data(year, weeks, df_sources)

    def process_data(self, game_data_worker):
        """Generates player info output data, such as age and position, for each player based on the collected data.

            Args:
                game_data_worker (SingleGameDataWorker): Processor for the current game, containing info on the roster, etc.

            Returns:
                pandas.DataFrame: Player info throughout this game. Indexed on Year, Week, Player ID, and Elapsed Time.

        """  # fmt: skip

        # Compute stats for each player on the team in this game
        list_of_player_dfs = game_data_worker.roster_df.reset_index().apply(
            self.__midgame_player_info,
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

    def __midgame_player_info(self, player_info, game_data_worker):
        """Determines the player info for one player throughout the game.

            Args:
                player_info (pandas.Series): Roster information for one player (number, ID, position, etc.).
                game_data_worker (SingleGameDataWorker): Processor for the current game, containing info on the roster, etc.

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
