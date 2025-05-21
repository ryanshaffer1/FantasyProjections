from __future__ import annotations

import pandas as pd

from config.player_id_config import PRIMARY_PLAYER_ID
from data_pipeline.features.feature_set import FeatureSet
from data_pipeline.utils.data_helper_functions import subsample_game_time


class StatsFeatureSet(FeatureSet):
    def __init__(self, features, sources):
        super().__init__(features, sources)
        self.pbp_df = None

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
            self.midgame_player_stats,
            args=(game_data_worker,),
            axis=1,
        )
        stats_df = pd.concat(list_of_player_dfs.tolist())

        # Set common index
        stats_df[["Year", "Week"]] = [game_data_worker.year, game_data_worker.week]
        stats_df = stats_df.reset_index().set_index(["Year", "Week", PRIMARY_PLAYER_ID, "Elapsed Time"])

        return stats_df

    def midgame_player_stats(self, player_info, game_data_worker):
        """Determines the mid-game stats for one player throughout the game.

            Args:
                player_info (pandas.Series): Roster information for one player (number, ID, position, etc.).
                game_times (list | str): Times in the game to output midgame stats for. May be a list of elapsed times in minutes,
                    in which case the soonest play-by-play time after that elapsed time will be used as the stats at that time. If a string is passed, no filtering occurs.

            Returns:
                pandas.DataFrame: Player's accumulated stat line at each time in the game. May have an additional (redundant) row for the final stat line.

        """  # fmt: skip

        # Set up dataframe covering player's contributions each play
        player_stats_df = pd.DataFrame()  # Output array
        # Game time elapsed
        player_stats_df["Elapsed Time"] = game_data_worker.pbp_df.reset_index()["Elapsed Time"]
        player_stats_df = player_stats_df.set_index("Elapsed Time")

        # Assign stats to player (e.g. passing yards, rushing TDs, etc.)
        player_stats_df = self.__assign_stats_from_plays(
            player_stats_df,
            game_data_worker.pbp_df,
            player_info,
            team_abbrev=player_info["Team"],
        )

        # Clean up the player dataframe
        player_stats_df = player_stats_df.drop(columns=["temp"])

        # Perform cumulative sum on all columns besides the list below
        for col in player_stats_df.columns:
            player_stats_df[col] = player_stats_df[col].cumsum()

        # Add some player info
        player_stats_df[PRIMARY_PLAYER_ID] = player_info[PRIMARY_PLAYER_ID]

        # Trim to just the game times of interest
        player_stats_df = subsample_game_time(player_stats_df, game_data_worker.game_times)

        return player_stats_df

    def __assign_stats_from_plays(self, player_stat_df, pbp_df, player_info, team_abbrev):
        # NOTE: 2-pt conversions should be excluded from stats
        """Parses play-by-play information to translate the result of plays into stats for the player currently being processed.

            Note that these stats are not cumulative, they only count stats for each play (later code outside this function accumulates the stats over the game).

            Args:
                player_stat_df (pandas.DataFrame): DataFrame containing some game info: elapsed time, as well as other relevant context (ex. Field Position).
                player_info (pandas.Series): Player information, including name, Player ID, position, etc.
                team_abbrev (str): Abbreviation for the team used in the play-by-play data

            Returns:
                pandas.DataFrame: Input player_stat_df DataFrame, with additional columns tracking whether the current player earned stats in each play.

        """  # fmt: skip

        player_id = player_info[PRIMARY_PLAYER_ID]

        # Passing Stats
        # Attempts (checks that selected player made the pass attempt)
        player_stat_df["Pass Att"] = pbp_df.apply(lambda x: (x["passer_player_id"] == player_id) & (x["sack"] == 0), axis=1)

        # Completions (checks that no interception was thrown and was not
        # marked incomplete)
        player_stat_df["Pass Cmp"] = pbp_df.apply(
            lambda x: ((x["complete_pass"] == 1) & (x["passer_player_id"] == player_id)),
            axis=1,
        )
        # Passing Yards
        player_stat_df["temp"] = pbp_df["passing_yards"]
        player_stat_df["Pass Yds"] = player_stat_df.apply(lambda x: x["temp"] if x["Pass Cmp"] else 0, axis=1)
        # Passing Touchdowns
        player_stat_df["temp"] = pbp_df["pass_touchdown"]
        player_stat_df["Pass TD"] = player_stat_df.apply(lambda x: x["Pass Cmp"] and (x["temp"] == 1), axis=1)
        # Interceptions
        player_stat_df["temp"] = pbp_df["interception"]
        player_stat_df["Int"] = player_stat_df.apply(lambda x: x["Pass Att"] and (x["temp"] == 1), axis=1)

        # Rushing Stats
        # Rush Attempts (checks that the selected player made the rush attempt)
        player_stat_df["Rush Att"] = pbp_df.apply(lambda x: (x["rusher_player_id"] == player_id), axis=1)
        # Rushing Yards
        player_stat_df["Rush Yds"] = pbp_df.apply(
            lambda x: x["rushing_yards"] if (x["rusher_player_id"] == player_id) else 0,
            axis=1,
        )
        # Rushing Touchdowns
        player_stat_df["Rush TD"] = pbp_df.apply(
            lambda x: x["td_player_id"] == player_id and (x["rush_touchdown"] == 1),
            axis=1,
        )

        # Receiving Stats
        # Receptions (checks that the selected player made the catch
        player_stat_df["Rec"] = pbp_df.apply(
            lambda x: (x["complete_pass"] == 1) & (x["receiver_player_id"] == player_id),
            axis=1,
        )
        # Receiving Yards
        player_stat_df["Rec Yds"] = pbp_df.apply(
            lambda x: x["receiving_yards"] if (x["complete_pass"] == 1) & (x["receiver_player_id"] == player_id) else 0,
            axis=1,
        )
        # Receiving Touchdowns
        player_stat_df["Rec TD"] = pbp_df.apply(
            lambda x: x["td_player_id"] == player_id and x["passer_player_id"] != player_id and (x["pass_touchdown"] == 1),
            axis=1,
        )

        # Misc. Stats
        # Add lateral receiving yards/rushing yards
        player_stat_df["Rec Yds"] = player_stat_df["Rec Yds"] + pbp_df.apply(
            lambda x: x["lateral_receiving_yards"] if x["lateral_receiver_player_id"] == player_id else 0,
            axis=1,
        )
        player_stat_df["Rush Yds"] = player_stat_df["Rush Yds"] + pbp_df.apply(
            lambda x: x["lateral_rushing_yards"] if x["lateral_rusher_player_id"] == player_id else 0,
            axis=1,
        )
        # Fumbles Lost
        player_stat_df["Fmb"] = pbp_df.apply(
            lambda x: (x["fumble_lost"] == 1)
            & (
                ((x["fumbled_1_player_id"] == player_id) & (x["fumble_recovery_1_team"] != team_abbrev))
                | ((x["fumbled_2_player_id"] == player_id) & (x["fumble_recovery_2_team"] != team_abbrev))
            ),
            axis=1,
        )

        # Replace all nan's with 0
        player_stat_df = player_stat_df.fillna(value=0)

        return player_stat_df
