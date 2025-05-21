from __future__ import annotations

import pandas as pd

from config.player_id_config import PRIMARY_PLAYER_ID
from data_pipeline.features.feature_set import FeatureSet
from data_pipeline.utils.data_helper_functions import subsample_game_time


class GameContextFeatureSet(FeatureSet):
    def __init__(self, features, sources):
        super().__init__(features, sources)
        self.pbp_df = None
        self.raw_rosters_df = None

    def collect_data(
        self,
        year: list[int] | range | int,
        weeks: list[int] | range,
        df_sources: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        super().collect_data(year, weeks, df_sources)
        self.pbp_df = next(iter(self.df_dict.values()))

    def process_data(self, game_data_worker):
        # Compute stats for each player on the team in this game
        list_of_player_dfs = game_data_worker.roster_df.reset_index().apply(
            self.midgame_game_context,
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

    def midgame_game_context(self, player_info, game_data_worker):
        """Determines the mid-game context for one player throughout the game.

            Args:
                player_info (pandas.Series): Roster information for one player (number, ID, position, etc.).
                game_times (list | str): Times in the game to output midgame stats for. May be a list of elapsed times in minutes,
                    in which case the soonest play-by-play time after that elapsed time will be used as the stats at that time. If a string is passed, no filtering occurs.

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
