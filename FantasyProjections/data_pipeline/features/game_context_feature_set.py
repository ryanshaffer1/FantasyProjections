"""Class used to collect/process/store data related to game context, such as teams, records, score, etc.

    Class:
        GameContextFeatureSet : Class that collects and processes data related to game context, including teams, records, scores, etc.

"""  # fmt: skip

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from config.player_id_config import PRIMARY_PLAYER_ID
from data_pipeline.features.feature_set import FeatureSet
from misc.data_helper_functions import subsample_game_time

if TYPE_CHECKING:
    from data_pipeline.features import Feature
    from data_pipeline.single_game_data_worker import SingleGameDataWorker


class GameContextFeatureSet(FeatureSet):
    """Class that collects and processes data related to game context, including teams, records, scores, etc.

        Sub-class of FeatureSet.

        Args:
            features (list[Feature]): All Feature (or sub-classes of Feature) objects to include in the feature set.
            sources (dict): Paths to both local and online sources to collect data associated with the feature set.

        Additional Class Attributes:
            thresholds (dict[str, list]): Maps each individual feature in the set to its normalization thresholds
            df_dict (dict): Stores loaded dataframes (including previously-cached dataframes) associated with each data source.
            pbp_df (pandas.DataFrame): Play-by-play data, as collected from data sources.

        Public Methods:
            collect_data : See FeatureSet
            process_data : Generates game context output data, such as team record and score, for each player based on the collected play-by-play data.

    """  # fmt: skip

    def __init__(self, features: list[Feature], sources: dict):
        """Constructor for GameContextFeatureSet objects.

            Args:
                features (list[Feature]): All Feature (or sub-classes of Feature) objects to include in the feature set.
                sources (dict): Paths to both local and online sources to collect data associated with the feature set.

        """  # fmt: skip

        super().__init__(features, sources)
        self.pbp_df = None

    def collect_data(
        self,
        year: int,
        weeks: list[int] | range,
        df_sources: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Collects data related to game context by searching the provided sources and checking for completeness.

            Modifies the attribute "pbp_df" to store the collected play-by-play data.

            Args:
                year (int): Year associated with the data (assumes the full year's worth of data is contained in one file).
                weeks (list | range): Weeks to ensure are included in the collected data (will search for them online if not).
                df_sources (dict, optional): Cached map of filenames to dataframes that have already been loaded (reduces re-loading). Defaults to None.

        """  # fmt: skip

        super().collect_data(year, weeks, df_sources)
        self.pbp_df = next(iter(self.df_dict.values()))

    def process_data(self, game_data_worker: SingleGameDataWorker) -> pd.DataFrame:
        """Generates game context output data, such as team record and score, for each player based on the collected play-by-play data.

            Args:
                game_data_worker (SingleGameDataWorker): Processor for the current game, containing info on the roster, etc.

            Returns:
                pandas.DataFrame: Game context info throughout this game. Indexed on Year, Week, Player ID, and Elapsed Time.

        """  # fmt: skip

        # Generate game context throughout the game for each player in this game
        list_of_player_dfs = game_data_worker.roster_df.reset_index().apply(
            self.__midgame_game_context,
            args=(game_data_worker,),
            axis=1,
        )
        stats_df = pd.concat(list_of_player_dfs.tolist())

        # Add some seasonal/weekly context to the dataframe
        stats_df["Team"] = game_data_worker.roster_df.loc[(stats_df[PRIMARY_PLAYER_ID], "Team")].tolist()
        stats_df["Opponent"] = game_data_worker.game_info.set_index("Team Abbrev").loc[stats_df["Team"], "Opp Abbrev"].to_list()
        stats_df["Site"] = game_data_worker.game_info.set_index("Team Abbrev").loc[stats_df["Team"], "Site"].to_list()
        stats_df[["Team Wins", "Team Losses", "Team Ties"]] = (
            game_data_worker.game_info.set_index("Team Abbrev")
            .loc[stats_df["Team"]][["Team Wins", "Team Losses", "Team Ties"]]
            .to_numpy()
        )
        stats_df[["Opp Wins", "Opp Losses", "Opp Ties"]] = (
            game_data_worker.game_info.set_index("Team Abbrev")
            .loc[stats_df["Opponent"]][["Team Wins", "Team Losses", "Team Ties"]]
            .to_numpy()
        )

        # Convert site and possession to 1/0
        stats_df["Site"] = pd.to_numeric(stats_df["Site"] == "Home")
        stats_df["Possession"] = pd.to_numeric(stats_df["Possession"])

        # Set common index
        stats_df[["Year", "Week"]] = [game_data_worker.year, game_data_worker.week]
        stats_df = stats_df.reset_index().set_index(["Year", "Week", PRIMARY_PLAYER_ID, "Elapsed Time"])

        return stats_df

    # PRIVATE METHODS

    def __midgame_game_context(self, player_info: pd.Series, game_data_worker: SingleGameDataWorker) -> pd.DataFrame:
        """Determines the mid-game context for one player throughout the game.

            Args:
                player_info (pandas.Series): Roster information for one player (number, ID, position, etc.).
                game_data_worker (SingleGameDataWorker): Processor for the current game, containing info on the roster, etc.

            Returns:
                pandas.DataFrame: Game context (e.g. score) at each time in the game. May have an additional (redundant) row for the final game time.

        """  # fmt: skip

        # Team sites
        game_site = game_data_worker.game_info.set_index("Team Abbrev").loc[player_info["Team"], "Site"].lower()
        opp_game_site = ["home", "away"][(["home", "away"].index(game_site) + 1) % 2]

        # Set up dataframe covering player's contributions each play
        player_stats_df = pd.DataFrame()  # Output array
        # Game time elapsed
        player_stats_df["Elapsed Time"] = game_data_worker.pbp_df.reset_index()["Elapsed Time"]
        player_stats_df = player_stats_df.set_index("Elapsed Time")
        # Possession
        player_stats_df["Possession"] = game_data_worker.pbp_df["posteam"] == player_info["Team"]
        # Field Position
        player_stats_df["Field Position"] = game_data_worker.pbp_df.apply(
            lambda x: x["yardline_100"] if (x["posteam"] == player_info["Team"]) else 100 - x["yardline_100"],
            axis=1,
        )
        # Score
        player_stats_df["Team Score"] = game_data_worker.pbp_df[f"total_{game_site}_score"]
        player_stats_df["Opp Score"] = game_data_worker.pbp_df[f"total_{opp_game_site}_score"]

        # Add some player info
        player_stats_df[PRIMARY_PLAYER_ID] = player_info[PRIMARY_PLAYER_ID]

        # Trim to just the game times of interest
        player_stats_df = subsample_game_time(player_stats_df, game_data_worker.game_times)

        return player_stats_df
