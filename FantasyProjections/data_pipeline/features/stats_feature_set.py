from __future__ import annotations

import logging

import pandas as pd

from config import data_files_config
from config.player_id_config import PRIMARY_PLAYER_ID, fill_blank_player_ids
from data_pipeline.features.feature_set import FeatureSet
from data_pipeline.stats_pipeline.scrape_pro_football_reference import scrape_box_score
from data_pipeline.utils import team_abbreviations as team_abbrs
from data_pipeline.utils.data_helper_functions import construct_game_id, subsample_game_time
from misc.manage_files import create_folders

# Set up logger
logger = logging.getLogger("log")


class StatsFeatureSet(FeatureSet):
    def __init__(self, features, sources):
        super().__init__(features, sources)
        self.pbp_df = None

    def collect_data(
        self,
        year: int,
        weeks: list[int] | range,
        df_sources: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        super().collect_data(year, weeks, df_sources)
        self.pbp_df = next(iter(self.df_dict.values()))

    def process_data(self, game_data_worker):
        # Compute stats for each player on the team in this game
        list_of_player_dfs = game_data_worker.roster_df.reset_index().apply(
            self.__midgame_player_stats,
            args=(game_data_worker,),
            axis=1,
        )
        stats_df = pd.concat(list_of_player_dfs.tolist())

        # Set common index
        stats_df[["Year", "Week"]] = [game_data_worker.year, game_data_worker.week]
        stats_df = stats_df.reset_index().set_index(["Year", "Week", PRIMARY_PLAYER_ID, "Elapsed Time"])

        return stats_df

    def collect_validation_data(
        self,
        final_stats_df: pd.DataFrame,
        aux_data_df: pd.DataFrame,
        scrape_missing=False,
        save_data=False,
    ):
        # Read saved truth data
        true_data_file = data_files_config.VALIDATION_FOLDER + f"{type(self).__name__}.csv"
        try:
            true_df = pd.read_csv(true_data_file).set_index([PRIMARY_PLAYER_ID, "Year", "Week"])
        except FileNotFoundError:
            logger.warning("True Stats data file not found! Building from scratch.")
            true_df = pd.DataFrame()
        # Compare saved truth data to input data to determine whether any saved truth data is missing
        missing_games, _ = self.__identify_missing_games(final_stats_df, true_df, fill_missing_data=False)

        # Add PFR ID to any players in the parsed database that are missing it
        final_stats_df = fill_blank_player_ids(
            players_df=final_stats_df,
            master_id_file=data_files_config.MASTER_PLAYER_ID_FILE,
            pfr_id_filename=data_files_config.PFR_ID_FILENAME,
            add_missing_pfr=scrape_missing,
            update_master=save_data,
        )

        if scrape_missing and len(missing_games) > 0:
            logger.info("Scraping web data:")
            missing_games_df = self.__scrape_missing_games(missing_games, aux_data_df)
            # Add primary player ID for all scraped players
            missing_games_df = (
                final_stats_df.reset_index()
                .set_index(["pfr_id", "Year", "Week"])
                .loc[:, [PRIMARY_PLAYER_ID]]
                .merge(missing_games_df, left_index=True, right_index=True)
                .reset_index()
                .set_index([PRIMARY_PLAYER_ID, "Year", "Week"])
            )
            true_df = pd.concat((true_df, missing_games_df))

            # Fill any missing data with 0 and log number of remaining missing players/games
            _, true_df = self.__identify_missing_games(final_stats_df, true_df, fill_missing_data=True)

            # Remove duplicates and format output df
            true_df = true_df.reset_index().drop_duplicates(keep="first").set_index([PRIMARY_PLAYER_ID, "Year", "Week"])
            # Save updated dataframe for use next time
            if save_data:
                create_folders(true_data_file)
                true_df.to_csv(true_data_file)
                logger.info(f"Saved data to {true_data_file}.")

        # Drop columns that don't get compared
        true_df = true_df.drop(columns=["pfr_id", "Player Name", "Team"])

        return true_df

    def __midgame_player_stats(self, player_info, game_data_worker):
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

    def __identify_missing_games(self, final_stats_df, true_df, fill_missing_data=False):
        final_stats_df = final_stats_df.copy()  # Copy df before manipulating it

        # Compare saved truth data to input data to determine whether any saved truth data is missing
        missing_players = final_stats_df.index.difference(true_df.index)
        # Gather unique game IDs that are missing
        missing_games = final_stats_df.loc[missing_players].reset_index()  # Filter to only the missing players
        missing_games["Game ID"] = missing_games.apply(construct_game_id, axis=1)
        missing_games = missing_games.drop_duplicates(subset=["Game ID"], keep="first").set_index("Game ID")
        logger.info(f"Truth Data missing {len(missing_players)} players from {len(missing_games)} games.")

        if fill_missing_data:
            logger.info("Assigning 0 to all missing player stats.")
            true_df = pd.concat((true_df, final_stats_df.loc[missing_players, ["Player Name", "Team"]]))
            with pd.option_context("future.no_silent_downcasting", True):
                true_df = true_df.fillna(0)

        return missing_games, true_df

    def __scrape_missing_games(self, missing_games, aux_data_df):
        missing_data_df = pd.DataFrame()
        # Scrape the corresponding stat URLs
        last_req_time = None
        for i, (game_id, row) in enumerate(missing_games.iterrows()):
            team_abbrevs = team_abbrs.convert_abbrev(
                row[["Team", "Opponent"]].to_list(),
                team_abbrs.pbp_abbrevs,
                team_abbrs.boxscore_website_abbrevs,
            )
            game_url = aux_data_df.drop_duplicates(subset=["PFR URL"]).loc[game_id, "PFR URL"]
            logger.info(f"({i + 1} of {len(missing_games)}) {game_id} from {game_url}")
            boxscore, last_req_time, success = scrape_box_score(game_url, team_abbrevs, last_req_time)

            if success:
                boxscore[["Year", "Week"]] = row.loc[["Year", "Week"]].to_list()
                boxscore = boxscore.set_index(["pfr_id", "Year", "Week"])
                missing_data_df = pd.concat((missing_data_df, boxscore))

        return missing_data_df
