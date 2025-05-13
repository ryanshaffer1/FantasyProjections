"""Creates and exports class to be used in NFL player statistics data collection.

    Classes:
        SingleGameDataWorker : Collects data (e.g. player stats or gambling odds) for a single NFL game. Automatically processes data upon initialization.
"""  # fmt: skip

import numpy as np
import pandas as pd

from config.player_id_config import ALT_PLAYER_IDS, PRIMARY_PLAYER_ID
from data_pipeline.utils.data_helper_functions import calc_game_time_elapsed, subsample_game_time


class SingleGameDataWorker:
    """Collects data (e.g. player stats or gambling odds) for a single NFL game. Automatically processes data upon initialization.

        Specific data processing steps are carried out in sub-classes.

        Args:
            seasonal_data (SeasonalDataCollector): "Parent" object containing data relating to the NFL season.
                Not stored as an object attribute.
            game_id (str): Game ID for specific game, as used by nfl-verse. Format is "{year}_{week}_{awayteam}_{home_team}", ex: "2021_01_ARI_TEN"
        Keyword Arguments:

        Additional Attributes Created during Initialization:
            year (int): Year of game being processed
            week (int): Week in NFL season of game being processed
            game_info (pandas.DataFrame): Information setting context for the game, including home/away teams, team records, etc.
            pbp_df (pandas.DataFrame): Play-by-play data for all plays in the game, taken from nfl-verse.
            roster_df (pandas.DataFrame): Players in the game (from both teams) to collect stats for.

        Public Methods:
            single_game_play_by_play : Filters and cleans play-by-play data for a specific game; keeps all plays from that game and sorts by increasing elapsed game time.

    """  # fmt: skip

    def __init__(self, seasonal_data, game_id, **kwargs):
        """Constructor for SingleGamePbpParser object.

            Args:
                seasonal_data (SeasonalDataCollector): "Parent" object containing data relating to the NFL season.
                    Not stored as an object attribute.
                game_id (str): Game ID for specific game, as used by nfl-verse. Format is "{year}_{week}_{awayteam}_{home_team}", ex: "2021_01_ARI_TEN"

            Additional Attributes Created during Initialization:
                year (int): Year of game being processed
                week (int): Week in NFL season of game being processed
                pbp_df (pandas.DataFrame): Play-by-play data for all plays in the game, taken from nfl-verse.
                roster_df (pandas.DataFrame): Players in the game (from both teams) to collect stats for.

        """  # fmt: skip

        # Optional keyword arguments
        self.game_times = kwargs.get("game_times", "all")

        # Re-format optional inputs
        if self.game_times != "all":
            self.game_times = np.array(self.game_times)

        # Basic info
        self.game_id = game_id
        self.year = seasonal_data.year
        self.week = int(self.game_id.split("_")[1])
        self.feature_sets = seasonal_data.feature_sets

        # Game info
        self.game_info = seasonal_data.all_game_info_df.loc[game_id]

        # Filter seasonal play-by-play database to just the plays in this game
        self.pbp_df = self.single_game_play_by_play(seasonal_data.pbp_df)

        # Roster info for this game from the two teams' seasonal data
        self.roster_df = (
            seasonal_data.all_rosters_df.loc[
                seasonal_data.all_rosters_df.index.intersection(
                    [(team, self.week) for team in self.pbp_df[["home_team", "away_team"]].iloc[0].to_list()],
                )
            ]
            .reset_index()
            .set_index(PRIMARY_PLAYER_ID)
        )

        # Process baseline data from inputs
        # Baseline data: Year, Week, Player ID, Elapsed Time
        self.midgame_df = self.build_baseline_df()

        # Add feature data to the dataframe
        for feature in self.feature_sets:
            feature_df = feature.process_data(self)
            self.midgame_df = pd.concat([self.midgame_df, feature_df], axis=1)

    # PUBLIC METHODS

    def build_baseline_df(self):
        # Collect all desired elapsed times for the game
        elapsed_time = subsample_game_time(self.pbp_df, self.game_times).index.to_list()

        # Generate row for each elapsed time for each player in the game
        midgame_df = pd.DataFrame(
            index=pd.MultiIndex.from_product([self.roster_df.index, elapsed_time], names=[PRIMARY_PLAYER_ID, "Elapsed Time"]),
        )
        # Add baseline data into new dataframe and set indices
        midgame_df["Year"] = self.year
        midgame_df["Week"] = self.week
        for col in [*ALT_PLAYER_IDS, "Player Name"]:
            midgame_df[col] = midgame_df.index.get_level_values(PRIMARY_PLAYER_ID).map(self.roster_df[col])
        midgame_df = midgame_df.reset_index().set_index(["Year", "Week", PRIMARY_PLAYER_ID, "Elapsed Time"])

        return midgame_df

    def single_game_play_by_play(self, pbp_df):
        """Filters and cleans play-by-play data for a specific game; keeps all plays from that game and sorts by increasing elapsed game time.

            Args:
                pbp_df (pandas.DataFrame): Play-by-play data for all plays in an NFL season, taken from nfl-verse.

            Returns:
                pandas.DataFrame: pbp_df input, filtered to only the plays with matching game_id. Elapsed Time is added as a column and set as the index.

        """  # fmt: skip

        # Make a copy of the input play-by-play df
        pbp_df = pbp_df.copy()

        # Filter to only the game of interest (using game_id)
        pbp_df = pbp_df[pbp_df["game_id"] == self.game_id]

        # Elapsed time
        pbp_df.loc[:, "Elapsed Time"] = pbp_df.apply(calc_game_time_elapsed, axis=1)

        # Sort by ascending elapsed time
        pbp_df = pbp_df.set_index("Elapsed Time").sort_index(ascending=True)

        return pbp_df
