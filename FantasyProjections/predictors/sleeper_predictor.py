"""Creates and exports classes and functions to be used as one approach to predicting NFL stats and Fantasy Football scores.

    Classes:
        SleeperPredictor : child of FantasyPredictor. Predicts NFL player stats using the Sleeper Fantasy web app/API.

    Functions (Interface points with Sleeper API):
        collect_sleeper_player_list : Pulls a list of all players in Sleeper database, and performs some initial processing of the list.
        collect_sleeper_projections : Pulls player projected stats from the Sleeper API for a set of weeks and adds to a pre-existing dict of projections by week.
"""  # fmt:skip

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pandas as pd
import torch
from sleeper_wrapper import Players, Stats

from config import data_files_config
from config.player_id_config import PRIMARY_PLAYER_ID
from data_pipeline.utils.name_matching import find_matching_name_ind
from misc.stat_utils import stats_to_fantasy_points
from predictors import FantasyPredictor

# Set up logger
logger = logging.getLogger("log")


@dataclass
class SleeperPredictor(FantasyPredictor):
    """Predictor of NFL players' stats in games, using the Sleeper Fantasy web app/API to generate predictions.

        Sub-class of FantasyPredictor.
        Pulls stat predictions from Sleeper Fantasy. How they compute their predictions is unclear.
        Sleeper API documentation: https://github.com/SwapnikKatkoori/sleeper-api-wrapper

        Args:
            name (str): name of the predictor object, used for logging/display purposes.
            player_id_file (str): filepath (including filename) to .json file storing all player/roster information from Sleeper. Required input.
            proj_dict_file (str, optional): filepath (including filename) to .json file storing all stat projections made by Sleeper.
                Defaults to None. If a file is not entered, or the file does not contain all the necessary data, updated information will
                automatically be requested from Sleeper.
            update_players (bool, optional): whether to request updated information on NFL players/rosters from Sleeper. Defaults to False.

        Additional Class Attributes:
            player_to_sleeper_id (dict): dictionary containing the names of all players in Sleeper's database,
                mapped to their Sleeper ID numbers (used in projections)
            all_proj_dict (dict): dictionary mapping NFL weeks (in "year-week" format) to the pre-game stat predictions from Sleeper for that week.
                Each set of pre-game stat predictions is a dict mapping a player's Sleeper ID to their predicted stat line.

        Public Methods:
            eval_model : Generates predicted stats for an input evaluation dataset, as provided by Sleeper.
            refresh_players : Updates player dictionary (player names to ID numbers) using Sleeper API

    """  # fmt:skip

    # CONSTRUCTOR
    player_id_file: str = data_files_config.MASTER_PLAYER_ID_FILE
    proj_dict_file: str | None = None
    update_players: bool = False

    def __post_init__(self):
        # Evaluates as part of the Constructor.
        # Generates attributes that are not simple data copies of inputs.

        # If no player dict file is input, player list must be updated from Sleeper API
        if self.player_id_file is None:
            self.update_players = True

        # Retrieve/update dataframe of players with their sleeper IDs
        self.player_ids = self.__gather_player_sleeper_ids()

        # Initialize attributes defined later (dependent on eval data used)
        self.all_proj_dict = {}

    # PUBLIC METHODS

    def eval_model(self, eval_data, **kwargs):
        """Generates predicted stats for an input evaluation dataset, as provided by Sleeper.

            Note that only pre-game predictions will be included in the evaluation result. If multiple game times in each game
            are present in eval_data, only one prediction per game will be made, with the other rows automatically dropped.

            Args:
                eval_data (StatsDataset): data to use for Neural Net evaluation (e.g. validation or test data).
                kwargs:
                    All keyword arguments are passed to the function stats_to_fantasy_points and to the PredictionResult constructor.
                    See the related documentation for descriptions and valid inputs. All keyword arguments are optional.

            Returns:
                PredictionResult: Object packaging the predicted and true stats together, which can be used for plotting,
                    performance assessments, etc.

        """  # fmt:skip

        # List of stats being used to compute fantasy score
        stat_columns = eval_data.y_data_columns

        # Remove duplicated games from eval data (only one projection per game from Sleeper)
        eval_data = eval_data.remove_game_duplicates()

        # Gather projections data from Sleeper API
        self.all_proj_dict = self.__gather_sleeper_proj(eval_data)

        # Build up array of predicted stats for all eval_data cases based on
        # sleeper projections dictionary
        stat_predicts = torch.empty(0)
        for _, id_row in eval_data.id_data.iterrows():
            year_week = id_row["Year-Week"]
            sleeper_id = int(id_row["sleeper_id"])
            if sleeper_id in self.player_ids["sleeper_id"].dropna().values:
                proj_stats = self.all_proj_dict[year_week][str(sleeper_id)]
                stat_line = torch.tensor(self.__reformat_sleeper_stats(proj_stats, stat_columns))
            else:
                stat_line = torch.zeros([len(stat_columns)])
            stat_predicts = torch.cat((stat_predicts, stat_line))

        # Compute fantasy points using stat lines (note that this ignores the
        # built-in fantasy points projection in the Sleeper API, which differs
        # from the sum of the stats)
        stat_predicts = stats_to_fantasy_points(
            torch.reshape(stat_predicts, [-1, len(stat_columns)]),
            stat_indices=stat_columns,
            **kwargs,
        )

        # True stats from eval data
        stat_truths = self.eval_truth(eval_data, **kwargs)

        # Create result object
        result = self._gen_prediction_result(stat_predicts, stat_truths, eval_data, **kwargs)

        return result

    def refresh_player_sleeper_ids(self, player_id_df, save_data=True):
        """Updates master list of player IDs using Sleeper API to fill in missing values for sleeper_id.

            It is not guaranteed that all missing IDs can be found, as Sleeper's player list/ID system are not comprehensive.
            Does not add extra players to the list, even if Sleeper has more unique players.

            Args:
                player_id_df (pandas.DataFrame): Master list of player IDs using various ID systems.
                save_data (bool, optional): Whether to save the updated ID list to the default file. Defaults to True.

            Returns:
                pandas.DataFrame: Input DataFrame modified to fill in whatever missing sleeper_id values can be found.

        """  # fmt:skip

        sleeper_player_df = collect_sleeper_player_list()

        # Drop players with missing primary IDs from df, and index on primary ID
        sleeper_player_df = sleeper_player_df.dropna(subset=PRIMARY_PLAYER_ID).set_index(PRIMARY_PLAYER_ID)

        # Update master list with the Sleeper ID of any players that are common to both DataFrames
        intersecting_ids = player_id_df.index.intersection(sleeper_player_df.index)
        player_id_df.loc[intersecting_ids, "sleeper_id"] = sleeper_player_df.loc[intersecting_ids, "sleeper_id"]

        # Add any more IDs that can be found via fuzzy name matching
        fuzzy_match_inds = (
            player_id_df[player_id_df["sleeper_id"].isna()]["Player Name"]
            .apply(find_matching_name_ind, args=(sleeper_player_df["full_name"],))
            .dropna()
        )
        player_id_df["sleeper_id"] = player_id_df.apply(
            lambda x: fuzzy_match_inds.loc[x.name] if x.name in fuzzy_match_inds.index else x["sleeper_id"],
            axis=1,
        )

        # Save data to master list
        if save_data and self.player_id_file is not None:
            player_id_df.to_csv(self.player_id_file)

        return player_id_df

    # PRIVATE METHODS

    def __gather_player_sleeper_ids(self):
        # Loads the master list of player IDs using different ID systems, and optionally updates it
        # with data from Sleeper.

        # Load Player IDs master list
        try:
            player_id_df = pd.read_csv(self.player_id_file, dtype={"sleeper_id": "Int64"}).set_index(PRIMARY_PLAYER_ID)
        except (ValueError, FileNotFoundError):
            player_id_df = collect_sleeper_player_list().rename(columns={"full_name": "Player Name"}).set_index(PRIMARY_PLAYER_ID)

        # Optionally use the Sleeper API to try and add missing Sleeper IDs to the master list
        if self.update_players:
            player_id_df = self.refresh_player_sleeper_ids(player_id_df)

        return player_id_df

    def __gather_sleeper_proj(self, eval_data):
        # Loads all_proj_dict from file (filename is an attribute of SleeperPredictor)
        # and checks if all the necessary data to evaluate against eval_data is present.
        # (i.e. are all the weeks in eval_data also present in all_proj_dict). If not,
        # updates all_proj_dict by requesting predictions for the missing weeks from Sleeper.

        # Unique year-week combinations in evaluation dataset
        eval_data.id_data["Year-Week"] = eval_data.id_data[["Year", "Week"]].astype(str).agg("-".join, axis=1)
        unique_year_weeks = list(eval_data.id_data["Year-Week"].unique())

        # Gather all stats from Sleeper
        if self.proj_dict_file is not None:
            try:
                with open(self.proj_dict_file, encoding="utf-8") as file:
                    all_proj_dict = json.load(file)
            except (FileNotFoundError, TypeError):
                logger.warning("Sleeper projection dictionary file not found during load process.")
                all_proj_dict = {}
        else:
            all_proj_dict = {}

        if not all(year_week in all_proj_dict for year_week in unique_year_weeks):
            # Gather any unsaved stats from Sleeper
            all_proj_dict = collect_sleeper_projections(all_proj_dict, unique_year_weeks)
            logger.info(f"Adding Year-Weeks to Sleeper projections dictionary: {self.proj_dict_file} \n{unique_year_weeks}")
            # Save projection dictionary to JSON file for use next time
            if self.proj_dict_file is not None:
                try:
                    with open(self.proj_dict_file, "w", encoding="utf-8") as file:
                        json.dump(all_proj_dict, file)
                except (FileNotFoundError, TypeError):
                    logger.warning("Sleeper projection dictionary file not found during save process.")

        return all_proj_dict

    def __reformat_sleeper_stats(self, stat_dict, stat_columns):
        # Re-names stats from Sleeper's format to the common names used across this project
        # and lists into the common stat line format.

        labels_df_to_sleeper = pd.read_csv(data_files_config.FEATURE_CONFIG_FILE, index_col=0)["sleeper"].dropna().to_dict()

        stat_line = []
        for stat in stat_columns:
            stat_value = stat_dict.get(labels_df_to_sleeper[stat], 0)
            stat_line.append(stat_value)

        return stat_line


# FUNCTIONS


def collect_sleeper_player_list():
    """Interfaces with Sleeper API to receive a list of all players in Sleeper database, and performs some initial processing of the list.

        Returns:
            pandas.DataFrame: DataFrame containing all players in Sleeper database, with sleeper_id column used to ID them.

    """  # fmt:skip

    # Call Sleeper API for player list and format into a DataFrame
    players = Players()
    sleeper_player_dict = players.get_all_players()
    sleeper_player_df = pd.DataFrame(sleeper_player_dict).transpose()
    sleeper_player_df = sleeper_player_df.sort_index().reset_index(drop=True).rename(columns={"player_id": "sleeper_id"})

    # Drop non-players from df (non-numeric sleeper IDs)
    sleeper_player_df = sleeper_player_df[sleeper_player_df["sleeper_id"].str.isnumeric()]

    # Convert Sleeper ID to integer (helpful in other methods, doesn't have to be this way though)
    sleeper_player_df["sleeper_id"] = sleeper_player_df["sleeper_id"].astype("Int64")

    return sleeper_player_df


def collect_sleeper_projections(all_proj_dict, year_weeks):
    """Pulls player projected stats from the Sleeper API for a set of weeks and adds to a pre-existing dict of projections by week.

        Args:
            all_proj_dict (dict): dictionary mapping NFL weeks (in "year-week" format) to the pre-game stat predictions from Sleeper for that week.
                Each set of pre-game stat predictions is a dict mapping a player's Sleeper ID to their predicted stat line.
            year_weeks (list): list of NFL weeks (in YYYY-w format, ex. "2024-1") to collect data for.
                If data is already in all_proj_dict, no action is taken. Otherwise, data is requested from the Sleeper API.

        Returns:
            all_proj_dict (dict): input dictionary, with any missing, requested weeks filled in with projections from the Sleeper API.

    """  # fmt:skip

    # Gather any unsaved stats from Sleeper
    stats = Stats()
    for year_week in year_weeks:
        if year_week not in all_proj_dict:
            [year, week] = year_week.split("-")
            week_proj = stats.get_week_projections("regular", year, week)
            all_proj_dict[year_week] = week_proj

    return all_proj_dict
