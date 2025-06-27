"""Creates and exports class to be used in NFL player statistics data collection.

    Classes:
        SeasonalDataCollector : Collects data (e.g. player stats) for all games in an NFL season. Automatically processes data upon initialization.
"""  # fmt: skip

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from config import data_files_config
from config.player_id_config import PRIMARY_PLAYER_ID
from misc.manage_files import create_folders
from misc.stat_utils import stats_to_fantasy_points

# Type checking imports
if TYPE_CHECKING:
    from data_pipeline.features.feature_set import FeatureSet

# Set up logger
logger = logging.getLogger("log")


@dataclass
class RosterFilter:
    filter_df: pd.DataFrame | None = None
    min_games_played: int = 3
    num_players: int = 300


class DatasetProcessor:
    def __init__(
        self,
        feature_sets: list[FeatureSet],
        midgame_df: pd.DataFrame,
        final_stats_df: pd.DataFrame,
        aux_data_df: pd.DataFrame,
        filter_df: pd.DataFrame | None = None,
        **kwargs,
    ):
        self.feature_sets = feature_sets
        self.midgame_df = midgame_df
        self.final_stats_df = final_stats_df
        self.aux_data_df = aux_data_df
        self.filter = RosterFilter(filter_df=filter_df, **kwargs)

    def generate_roster_filter(self, rosters_df, save_file=None, plot_filter=False):
        """Generates a short list of players to focus data collection on, based on highest average Fantasy Points per game.

            Rules
            1. "Currently" active players only (active at some point in the last season being processed)
            2. Sort by fantasy points per game played
            3. Player must have played in at least 5 games
            4. Take the top num_players number of players per criteria 2

            Saves filtered list of players to a csv file to be used for later data collection.

            Args:
                rosters_df (pandas.DataFrame): DataFrame containing all weekly NFL rosters for a given series. Loaded from nfl-verse
                save_file (str, optional): csv file path to save filtered player list. Defaults to None (file is not saved).
                plot_filter (bool, optional): Whether to visualize the filtered player list by position and average Fantasy Points. Defaults to False.

            Returns:
                pandas.DataFrame: List of players to include in the filter, along with some additional data like team, position, average Fantasy Points, etc.

        """  # fmt: skip

        # Add Fantasy Points to final stats if not already computed
        self.final_stats_df = stats_to_fantasy_points(self.final_stats_df)

        # 1. Active players only
        # Removes 'RET','CUT','DEV', 'TRC' (I think means free agent?)
        active_statuses = ["ACT", "INA", "RES", "EXE"]
        rosters_df = rosters_df[rosters_df.apply(lambda x: x["status"] in active_statuses, axis=1)]
        # 1b. Remove tracking of week-by-week status - only one row per player.
        # Take the entry from the last week they've played
        last_week_played = rosters_df.loc[:, [PRIMARY_PLAYER_ID, "week"]].groupby([PRIMARY_PLAYER_ID]).max()
        filter_df = rosters_df[
            rosters_df.apply(
                lambda x: (x[PRIMARY_PLAYER_ID] in last_week_played.index.to_list())
                & (x["week"] == last_week_played.loc[x[PRIMARY_PLAYER_ID], "week"]),
                axis=1,
            )
        ]

        # 2. Add average fantasy points per game to df
        fantasy_avgs = (
            self.final_stats_df.reset_index()
            .loc[:, [PRIMARY_PLAYER_ID, "Fantasy Points"]]
            .groupby([PRIMARY_PLAYER_ID])
            .mean()
            .rename(columns={"Fantasy Points": "Fantasy Avg"})
        )
        filter_df = filter_df.merge(right=fantasy_avgs, on=PRIMARY_PLAYER_ID)

        # 3. Count games played - instances of player name
        game_counts = self.final_stats_df.reset_index()[PRIMARY_PLAYER_ID].value_counts()
        filter_df = filter_df[filter_df[PRIMARY_PLAYER_ID].apply(lambda x: game_counts[x] >= self.filter.min_games_played)]

        # 4a. Sort by max avg fantasy points
        filter_df = filter_df.sort_values(by=["Fantasy Avg"], ascending=False).reset_index(drop=True)
        # 4b. Take first x (num_players) players
        filter_df = filter_df.iloc[0 : self.filter.num_players]

        # Clean up df for saving
        filter_df = filter_df[[PRIMARY_PLAYER_ID, "full_name", "Fantasy Avg", "team", "position", "jersey_number"]]
        filter_df = filter_df.rename(
            columns={
                "team": "Team",
                "position": "Position",
                "jersey_number": "Number",
                "full_name": "Name",
                PRIMARY_PLAYER_ID: "Player ID",
            },
        )

        # Save
        if save_file:
            filter_df.to_csv(save_file)
            logger.info(f"Saved Roster Filter to {save_file}")

        # Print data and breakdown by team/position
        logger.info("Roster Filter Breakdown by Team:")
        logger.info(f"{filter_df['Team'].value_counts()}")
        logger.info("Roster Filter Breakdown by Position:")
        logger.info(f"{filter_df['Position'].value_counts()}")

        if plot_filter:
            self.__create_filter_plot()

        self.filter.filter_df = filter_df

    def apply_roster_filter(self):
        """Trims previously-generated NFL stats DataFrames (midgame and final stats) to only include players in a filtered list.

            Args:

            Returns:
                pandas.DataFrame: midgame_df, trimmed to only include the players in filter_df.
                pandas.DataFrame: final_stats_df, trimmed to only include the players in filter_df.

        """  # fmt: skip
        if self.filter.filter_df is None:
            logger.warning("No filter_df provided. Skipping roster filter application.")
            return

        filter_ids = self.filter.filter_df["Player ID"].to_list()
        self.midgame_df = self.midgame_df[self.midgame_df.index.to_frame()[PRIMARY_PLAYER_ID].apply(lambda x: x in filter_ids)]
        self.final_stats_df = self.final_stats_df[self.final_stats_df.apply(lambda x: x.name[0] in filter_ids, axis=1)]

    def validate_final_df(self, **kwargs):
        # Optional input to save data
        save_data = kwargs.get("save_data", False)

        # Collect truth data from feature sets
        all_truth_data = pd.DataFrame()
        all_truth_columns = []
        for feature_set in self.feature_sets:
            val_df = feature_set.collect_validation_data(
                final_stats_df=self.final_stats_df,
                aux_data_df=self.aux_data_df,
                **kwargs,
            )
            all_truth_data = pd.concat((all_truth_data, val_df))
            all_truth_columns.extend(val_df.columns.tolist())

        # Perform comparison of the two dataframes
        diff_df = self.__compare_dfs(all_truth_data, self.final_stats_df, all_truth_columns)

        # Log comparison performance
        self.__print_validation_comparison(diff_df, all_truth_columns)

        # Save validation results to file if requested
        if save_data:
            create_folders(data_files_config.PARSING_VALIDATION_FILE)
            diff_df.to_csv(data_files_config.PARSING_VALIDATION_FILE)

        # Plot differences in numerical values
        self.__plot_validation_comparison(diff_df, all_truth_columns)

    def __compare_dfs(self, true_df, final_stats_df, columns):
        # Merge dataframes and compute differences in statistics
        merged_df = true_df.merge(final_stats_df, how="inner", left_index=True, right_index=True)
        for col in columns:
            merged_df[f"{col}_diff"] = merged_df[f"{col}_y"] - merged_df[f"{col}_x"]  # Estimated minus truth
        diff_df = merged_df[["Player Name"] + [col + "_diff" for col in columns]]
        return diff_df

    def __print_validation_comparison(self, diff_df, columns):
        # Log comparison performance
        col_diffs = diff_df[[col + "_diff" for col in columns]]
        avg_diffs = col_diffs.mean()
        num_nonzero = col_diffs.astype(bool).sum()
        logger.info(
            f"Number of differences between parsed and true data: {num_nonzero.sum()} ({100 * num_nonzero.sum() / col_diffs.size:.2f}%)",
        )
        logger.debug(f"Number of differences by stat: \n{num_nonzero}")
        logger.debug(f"Average difference by stat: \n{avg_diffs}")

    def __plot_validation_comparison(self, diff_df, columns):
        x = [list(range(len(columns))) for _ in range(diff_df.shape[0])]
        y = diff_df[[s + "_diff" for s in columns]].stack()

        _, ax = plt.subplots(1, 1)
        ax.scatter(x, y)
        ax.set_ylabel("Difference, Truth - Parsed Data")
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns)
        for label in ax.get_xticklabels():
            label.set(rotation=45, horizontalalignment="right")
        ax.set_title("Play-By-Play Parsing Validation vs. True Statlines")

        plt.show(block=False)

    def __create_filter_plot(self):
        """Plots bar chart of average Fantasy Points for each player in filtered list (sorted in descending order), colored by position.

            Args:
                filter_df (pandas.DataFrame): List of players included in the filter, along with some additional data like team, position, average Fantasy Points, etc.
                num_players (int): Number of players to include in list.

        """  # fmt: skip
        if self.filter.filter_df is None:
            logger.warning("No filter_df provided. Skipping filter plot creation.")
            return

        # Plot bar chart of points by rank, colored by position
        position_colors = {"QB": "tab:blue", "RB": "tab:orange", "WR": "tab:green", "TE": "tab:purple", "Other": "tab:brown"}
        plot_colors = self.filter.filter_df["Position"].apply(
            lambda x: position_colors[x] if x in position_colors else position_colors["Other"],
        )
        if position_colors["Other"] not in plot_colors:
            del position_colors["Other"]
        ax = plt.subplots(1, 1)[1]
        ax.bar(
            range(1, self.filter.num_players + 1),
            self.filter.filter_df["Fantasy Avg"].to_list(),
            width=1,
            color=plot_colors,
            linewidth=0,
        )
        plt.xlabel("Rank")
        plt.ylabel("Avg. Fantasy Score")
        plt.title("Fantasy Performance of Filtered Player List")

        def lp(i):
            return ax.plot([], color=position_colors[i], label=i)[0]

        leg_handles = [lp(i) for i in position_colors]
        plt.legend(handles=leg_handles)
        plt.show()
