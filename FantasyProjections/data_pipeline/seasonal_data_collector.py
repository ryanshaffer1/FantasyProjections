"""Creates and exports class to be used in NFL player statistics data collection.

    Classes:
        SeasonalDataCollector : Collects data (e.g. player stats) for all games in an NFL season. Automatically processes data upon initialization.
"""  # fmt: skip

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import dateutil.parser as dateparse
import pandas as pd

from config.player_id_config import PLAYER_IDS, PRIMARY_PLAYER_ID, fill_blank_player_ids
from data_pipeline.single_game_data_worker import SingleGameDataWorker
from data_pipeline.utils import team_abbreviations as team_abbrs
from data_pipeline.utils.data_helper_functions import (
    clean_team_names,
    compute_team_record,
)
from misc.manage_files import collect_input_dfs

# Type checking imports
if TYPE_CHECKING:
    from data_pipeline.features.feature_set import FeatureSet

# Set up logger
logger = logging.getLogger("log")


class SeasonalDataCollector:
    """Collects data (e.g. player stats) for all games in an NFL season. Automatically processes data upon initialization.

        Specific data processing steps are carried out in sub-classes.

        Args:
            year (int): Year for season (e.g. 2023).
            team_names (str | list, optional): Either "all" or a list of full team names (e.g. ["Arizona Cardinals", "Baltimore Ravens", ...]). Defaults to "all".
            weeks (list, optional): Weeks in the NFL season to collect data for. Defaults to range(1,19).

        Keyword Arguments:
            filter_df (pandas.DataFrame, optional): Filter for roster, i.e. the list of players to collect data for. Defaults to None (collect data for every player).
                Not stored as an object attribute.

        Additional Attributes Created during Initialization:
            pbp_df (pandas.DataFrame): Play-by-play data for all plays in an NFL season, taken from nfl-verse.
            raw_rosters_df (pandas.DataFrame): Weekly roster information for all teams in an NFL season, taken from nfl-verse.
            all_rosters_df (pandas.DataFrame): Filtered roster dataframe to include only players of interest (skill positions, in the optional filter_df, etc.)
            games (list): List of SingleGameDataWorker (or sub-class) objects containing data for every game in the NFL season.


        Objects Created:
            List of SingleGameDataWorker (or sub-class) objects

        Public Methods:
            gather_all_game_data : Concatenates all relevant data from individual games in self.games into larger DataFrames for the full season.
            process_rosters : Trims DataFrame of all NFL week-by-week rosters in a given year to include only players of interest and data columns of interest.

    """  # fmt: skip

    def __init__(
        self,
        data_files_config: dict,
        year: int,
        feature_sets: list[FeatureSet],
        team_names: list[str] | str = "all",
        weeks: list[int] | range | None = None,
        **kwargs,
    ):
        """Constructor for SeasonalDataCollector class.

            Args:
                year (int): Year for season (e.g. 2023).
                team_names (str | list, optional): Either "all" or a list of full team names (e.g. ["Arizona Cardinals", "Baltimore Ravens", ...]). Defaults to "all".
                weeks (list, optional): Weeks in the NFL season to collect data for. Defaults to range(1,19).
                kwargs:
                    filter_df (pandas.DataFrame, optional): Filter for roster, i.e. the list of players to collect data for. Defaults to None (collect data for every player).
                        Not stored as an object attribute.

            Additional Attributes Created during Initialization:
                pbp_df (pandas.DataFrame): Play-by-play data for all plays in an NFL season, taken from nfl-verse.
                raw_rosters_df (pandas.DataFrame): Weekly roster information for all teams in an NFL season, taken from nfl-verse.
                all_rosters_df (pandas.DataFrame): Filtered roster dataframe to include only players of interest (skill positions, in the optional filter_df, etc.)
                games (list): List of SingleGameDataWorker (or sub-class) objects containing data for every game in the NFL season.

        """  # fmt: skip

        # Handle unspecified weeks: all weeks
        if weeks is None:
            weeks = list(range(1, 19))

        # Optional keyword arguments
        game_times = kwargs.get("game_times", "all")
        filter_df = kwargs.get("filter_df")

        # Basic attributes
        self.data_files_config = data_files_config
        self.year = year
        self.weeks = weeks
        self.team_names = clean_team_names(team_names, self.year)
        self.feature_sets = feature_sets

        # Collect input data
        dfs_dict, df_sources = collect_input_dfs(
            self.year,
            self.weeks,
            self.data_files_config["local_file_paths"],
            self.data_files_config["online_file_paths"],
            online_avail=True,
        )
        self.pbp_df = dfs_dict[0].pop("pbp")
        self.raw_rosters_df = dfs_dict[0].pop("roster")

        # Gather team rosters based on input teams and weeks and optional filter
        self.all_rosters_df = self.process_rosters(filter_df)

        # Process team records, game sites, and other game info for every game of the year, for every team
        self.all_game_info_df = self.get_game_info()

        # Collect input data for all seasonal features
        for feature_set in self.feature_sets:
            feature_set.collect_data(year, weeks, df_sources)

        # List of SingleGameData objects
        self.games = self.generate_games(game_times)

        # Gather all midgame stats from the individual games
        self.midgame_df, *_ = self.gather_all_game_data(["midgame_df"])
        # Highlight final stats
        self.final_stats_df = self.gather_final_stats()

    # PUBLIC METHODS
    def gather_final_stats(self):
        # Remove rows with duplicated year/week/player, keeping only the last elapsed time
        duplicates = self.midgame_df.index.to_frame().duplicated(subset=["Year", "Week", PRIMARY_PLAYER_ID], keep="last")
        final_stats_df = self.midgame_df[~duplicates]

        # Drop the Elapsed Time column
        final_stats_df = final_stats_df.reset_index().drop(columns=["Elapsed Time"])

        # Index on just Player ID
        final_stats_df = final_stats_df.set_index([PRIMARY_PLAYER_ID, "Year", "Week"])

        return final_stats_df

    def generate_games(self, game_times):
        """Creates a SingleGamePbpParser object for each unique game included in the SeasonalDataCollector.

                Args:
                    game_times (list | str): Elapsed time steps to save data for (e.g. every minute of the game, every 5 minutes, etc.). May be "all", meaning every play.

                Returns:
                    list: List of SingleGamePbpParser objects containing data for each game in the SeasonalDataCollector

            """  # fmt: skip

        team_abbrevs_to_process = [team_abbrs.pbp_abbrevs[name] for name in self.team_names]

        # Generate list of game IDs to process (based on weeks and teams to include)
        game_ids = []
        for game_id in self.all_game_info_df.index.unique():
            game_info = self.all_game_info_df.loc[game_id].iloc[0]
            week = int(game_info["Week"])
            home_team = game_info["Home Team Abbrev"]
            away_team = game_info["Away Team Abbrev"]
            if week in self.weeks and (home_team in team_abbrevs_to_process or away_team in team_abbrevs_to_process):
                game_ids.append(game_id)

        # Create objects that process stats for each game of interest
        games = []
        n_games = len(game_ids)
        logger.info(f"Processing {n_games} games:")
        for i, game_id in enumerate(game_ids):
            logger.info(f"({i + 1} of {n_games}): {game_id}")
            # Process data for single game
            game = SingleGameDataWorker(self, game_id, game_times=game_times)

            # Add to list of games
            games.append(game)

        return games

    def gather_all_game_data(self, df_fields: list[str] | str) -> tuple[pd.DataFrame, ...]:
        """Concatenates all relevant data from individual games in self.games into larger DataFrames for the full season.

            Args:
                df_fields (list | str): Names of the DataFrame properties in each game to concatenate.

            Returns:
                tuple(pandas.DataFrame): DataFrames of data consolidated across all games, one for each element in df_fields.

        """  # fmt: skip

        # Handle single field passed as string
        if isinstance(df_fields, str):
            df_fields = [df_fields]

        dfs = [pd.DataFrame()] * len(df_fields)
        for i, df_field in enumerate(df_fields):
            # Initialize empty dataframe
            game_data_df = pd.DataFrame()
            for game in self.games:
                game_data_df = pd.concat((game_data_df, getattr(game, df_field)))

            logger.debug(f"{self.year} {df_field} rows: {game_data_df.shape[0]}")
            dfs[i] = game_data_df

        return tuple(dfs)

    def process_rosters(self, filter_df=None):
        """Trims DataFrame of all NFL week-by-week rosters in a given year to include only players of interest and data columns of interest.

            Args:
                filter_df (pandas.DataFrame, optional): Pre-determined list of players to include. Defaults to None.

            Attributes Modified:
                pandas.DataFrame: all_rosters_df filtered to players of interest, several columns removed, and indexed on Team & Week

        """  # fmt: skip

        # Copy of object attribute
        all_rosters_df = self.raw_rosters_df.copy()

        # Filter to only the desired weeks
        all_rosters_df = all_rosters_df[all_rosters_df.apply(lambda x: x["week"] in self.weeks, axis=1)]

        # Filter to only the desired teams
        all_rosters_df = all_rosters_df[
            all_rosters_df.apply(lambda x: x["team"] in [team_abbrs.pbp_abbrevs[name] for name in self.team_names], axis=1)
        ]

        # Optionally filter based on subset of desired players
        if filter_df is not None:
            all_rosters_df = all_rosters_df[all_rosters_df.apply(lambda x: x["full_name"] in filter_df["Name"].to_list(), axis=1)]

        # Filter to only skill positions
        # Positions currently being tracked for stats
        skill_positions = ["QB", "RB", "FB", "HB", "WR", "TE"]
        all_rosters_df = all_rosters_df[all_rosters_df.apply(lambda x: x["position"] in skill_positions, axis=1)]

        # Filter to only active players
        valid_statuses = ["ACT"]
        all_rosters_df = all_rosters_df[all_rosters_df.apply(lambda x: x["status"] in valid_statuses, axis=1)]

        # Compute age based on birth date. Assign birth year of 2000 for anyone with missing birth date...
        all_rosters_df["Age"] = all_rosters_df["season"] - all_rosters_df["birth_date"].apply(
            lambda x: dateparse.parse(x).year if isinstance(x, str) else 2000,
        )

        # Trim to just the fields that are useful
        all_rosters_df = all_rosters_df[[*PLAYER_IDS, "team", "week", "position", "jersey_number", "full_name", "Age"]]
        # Reformat
        all_rosters_df = (
            all_rosters_df.rename(
                columns={
                    "team": "Team",
                    "week": "Week",
                    "position": "Position",
                    "jersey_number": "Number",
                    "full_name": "Player Name",
                },
            )
            .set_index(["Team", "Week"])
            .sort_index()
        )

        # Update player IDs
        all_rosters_df = fill_blank_player_ids(
            players_df=all_rosters_df,
            pfr_player_url_intro=self.data_files_config["pfr_player_url_intro"],
            master_id_file=self.data_files_config["master_player_id_file"],
            pfr_id_filename=self.data_files_config["pfr_id_filename"],
            add_missing_pfr=False,
            update_master=False,
        )

        return all_rosters_df

    def get_game_info(self):
        """Generates info on every game for each team in a given year: who is home vs away, and records of each team going into the game.

            Also generates url to get game stats from pro-football-reference.com.
            Note that each game is included twice - once from the perspective of each team (i.e. "Team" and "Opponent" info are swapped).

            Attributes Modified:
                all_game_info_df: DataFrame containing game info on each unique game in the NFL season being processed (teams, pre-game team records, etc.)
        """  # fmt: skip

        pbp_df = self.pbp_df.copy()

        # Filter df to only the final play of each game
        pbp_df = pbp_df.drop_duplicates(subset="game_id", keep="last")

        # Filter to only regular-season games
        pbp_df = pbp_df[pbp_df["season_type"] == "REG"]

        # Output data frame
        all_game_info_df = pd.DataFrame()

        for team_abbr in team_abbrs.pbp_abbrevs.values():
            # Filter to only games including the team of interest
            scores_df = pbp_df.copy()
            scores_df = scores_df[(scores_df["home_team"] == team_abbr) | (scores_df["away_team"] == team_abbr)]

            # Track team name and opponent name
            scores_df["Team Abbrev"] = team_abbr
            scores_df["Opp Abbrev"] = scores_df.apply(
                lambda x: x["away_team"] if x["home_team"] == x["Team Abbrev"] else x["home_team"],
                axis=1,
            )

            # Track game site and home/away team
            scores_df["Site"] = scores_df.apply(lambda x: "Home" if x["home_team"] == x["Team Abbrev"] else "Away", axis=1)
            scores_df = scores_df.rename(
                columns={"week": "Week", "home_team": "Home Team Abbrev", "away_team": "Away Team Abbrev"},
            )

            # URL to get stats from
            scores_df["game_date"] = scores_df.apply(lambda x: dateparse.parse(x["game_date"]).strftime("%Y%m%d"), axis=1)
            scores_df["PFR URL"] = scores_df.apply(
                lambda x: self.data_files_config["pfr_boxscore_url_intro"]
                + x["game_date"]
                + "0"
                + team_abbrs.convert_abbrev(
                    x["Home Team Abbrev"],
                    team_abbrs.pbp_abbrevs,
                    team_abbrs.roster_website_abbrevs,
                )
                + ".htm",
                axis=1,
            )

            # Track ties, wins, and losses
            scores_df = compute_team_record(scores_df)

            # Remove unnecessary columns
            columns_to_keep = [
                "Week",
                "Team Abbrev",
                "Opp Abbrev",
                "game_id",
                "PFR URL",
                "Site",
                "Home Team Abbrev",
                "Away Team Abbrev",
                "Team Wins",
                "Team Losses",
                "Team Ties",
            ]
            scores_df = scores_df[columns_to_keep].set_index(["Week", "Team Abbrev"]).sort_index()

            # Append to dataframe of all teams' games
            all_game_info_df = pd.concat([all_game_info_df, scores_df])

        # Clean up df for output
        all_game_info_df = all_game_info_df.reset_index().set_index(["game_id"]).sort_index()
        all_game_info_df["Team Name"] = all_game_info_df["Team Abbrev"].apply(
            lambda x: team_abbrs.invert(team_abbrs.pbp_abbrevs)[x],
        )

        return all_game_info_df
