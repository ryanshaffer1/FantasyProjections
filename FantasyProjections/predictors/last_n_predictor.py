"""Creates and exports classes to be used as various approaches to predicting NFL stats and Fantasy Football scores.

    Classes:
        LastNPredictor : child of FantasyPredictor. Predicts NFL player stats using an average of the player's stats in recent games.
"""

from dataclasses import dataclass

import numpy as np
import torch
from misc.stat_utils import stats_to_fantasy_points
from predictors import FantasyPredictor


@dataclass
class LastNPredictor(FantasyPredictor):
    """Predictor of NFL players' stats in games, using an average of the player's stats in recent games.

        Sub-class of FantasyPredictor.

        Args:
            name (str): name of the predictor object, used for logging/display purposes.
            n (int, optional): number of previous games to average over. Defaults to 1.

        Public Methods:
            eval_model : Generates predicted stats for an input evaluation dataset, based on an average of the player's stats in recent games.
    """

    # CONSTRUCTOR
    n: int = 1


    # PUBLIC METHODS

    def eval_model(self, eval_data, all_data=None, **kwargs):
        """Generates predicted stats for an input evaluation dataset, based on an average of the player's stats in recent games.

            Note that only pre-game predictions will be included in the evaluation result. If multiple game times in each game
            are present in eval_data, only one prediction per game will be made, with the other rows automatically dropped.

            Args:
                eval_data (StatsDataset): data to use for Predictor evaluation (e.g. validation or test data).
                all_data (StatsDataset, optional): every game that can be used as a lookup for previous games within eval_data.
                    Since eval_data may not include a player's whole career, all_data is intended to provide a broader search space
                    to find a player's previous game. Defaults to eval_data.

            Keyword-Args:
                All keyword arguments are passed to the function stats_to_fantasy_points and to the PredictionResult constructor.
                See the related documentation for descriptions and valid inputs.
                All keyword arguments are optional.

            Returns:
                PredictionResult: Object packaging the predicted and true stats together, which can be used for plotting,
                    performance assessments, etc.
        """

        # Optional all_data input
        if all_data is None:
            all_data = eval_data

        # Override default kwargs to stats_to_fantasy_points
        kwargs["normalized"] = kwargs.get("normalized",True)

        # Drop all the duplicated rows that are for the same game, and only
        # dependent on elapsed game time - that variable is irrelevant here, so we
        # can greatly simplify
        eval_data = eval_data.remove_game_duplicates()
        all_data = all_data.remove_game_duplicates()

        # For every row in all_data, find the index in all_data that contains the previous game played by the same player
        all_ids = self.__link_previous_games(all_data)

        # Get the stats for each previous game, in order
        all_ids["tensor"] = self.__stats_from_past_games(all_ids, all_data.y_data, n=self.n)

        # Grab stats for each game in the evaluation data
        prev_game_stats_df = eval_data.id_data.apply(
            lambda x: all_ids.loc[(x["Player ID"], x["Year"], x["Week"]), "tensor"], axis=1)
        prev_game_stats = torch.tensor(prev_game_stats_df.to_list())
        # List of stats being used to compute fantasy score
        stat_columns = eval_data.y_data_columns

        # Un-normalize and compute Fantasy score
        stat_predicts = stats_to_fantasy_points(
            prev_game_stats, stat_indices=stat_columns, **kwargs)
        # True stats from eval data
        stat_truths = self.eval_truth(eval_data, **kwargs)

        # Create result object
        result = self._gen_prediction_result(stat_predicts, stat_truths, eval_data, **kwargs)

        return result

    # PRIVATE METHODS

    def __link_previous_games(self, all_data):
        # For every row in all_data (note that each row corresponds to a unique player/game),
        # find the index in all_data that contains the previous game played by the same player

        # Variables needed to search for previous games and convert stats to
        # fantasy points
        first_year_in_dataset = min(all_data.id_data["Year"])
        # Set up dataframe to use to search for previous games
        all_ids = all_data.id_data.copy()[["Player ID", "Year", "Week"]]
        all_ids = all_ids.reset_index().set_index(
            ["Player ID", "Year", "Week"]).sort_index()
        indices = all_ids.index
        all_ids["Player ID Copy"] = all_ids.index.get_level_values(0).to_list()
        all_ids["Prev Year"] = all_ids.index.get_level_values(1).to_list()
        all_ids["Prev Week"] = all_ids.index.get_level_values(2).to_list()
        all_ids["Prev Game Found"] = False
        all_ids["Continue Prev Search"] = True
        while any(all_ids["Continue Prev Search"]):
            # Only make changes to rows of the dataframe that are continuing the search
            search_ids = all_ids["Continue Prev Search"]
            # Find the next (previous) week to look for
            all_ids.loc[search_ids,"Prev Week"] = all_ids.loc[search_ids,"Prev Week"] - 1
            # If previous week is set to Week 0, fix this by going to Week 18 of the
            # previous year
            all_ids.loc[search_ids, "Prev Year"] = all_ids.loc[search_ids].apply(
                lambda x: x["Prev Year"] - 1 if x["Prev Week"] == 0 else x["Prev Year"], axis=1)
            all_ids.loc[search_ids, "Prev Week"] = all_ids.loc[search_ids].apply(
                lambda x: 18 if x["Prev Week"] == 0 else x["Prev Week"], axis=1)

            # Check if previous game is in the dataframe
            all_ids.loc[search_ids, "Prev Game Found"] = all_ids.loc[search_ids].apply(
                lambda x: (x["Player ID Copy"], x["Prev Year"], x["Prev Week"]) in indices, axis=1)

            # Check whether to continue searching for a previous game for this player
            all_ids.loc[search_ids,"Continue Prev Search"] = np.logical_not(
                all_ids.loc[search_ids,"Prev Game Found"]) & (
                    all_ids.loc[search_ids,"Prev Year"] >= first_year_in_dataset)

        # Assign the index (row) of the previous game
        all_ids["Prev Game Index"] = np.nan
        valid_rows = all_ids["Prev Game Found"]
        all_ids.loc[valid_rows, "Prev Game Index"] = all_ids.loc[valid_rows].apply(
            lambda x: all_ids.loc[(x["Player ID Copy"], x["Prev Year"], x["Prev Week"]), "index"], axis=1)

        return all_ids


    def __stats_from_past_games(self,all_ids, y_data, n=1):
        # Collect game stats from y_data over the last n games
        # Where all_ids contains the "linked list" to previous games

        answer = []
        sorted_index = all_ids.sort_values(by=["index"]).index
        for row_ind in all_ids.index:
            array = []
            curr_row = all_ids.loc[row_ind] # Moving tracker of the data row
            for _ in range(n):
                prev_game_id = curr_row["Prev Game Index"]
                if prev_game_id >= 0:
                    # Grab stats from the previous-game's row
                    array.append(y_data[int(prev_game_id)])
                    # Move onto the next-previous row
                    curr_row = all_ids.loc[sorted_index[int(prev_game_id)]]
                else:
                    break

            # Return average game stats across n games (if any games were found)
            if array:
                answer.append(list(np.mean(array,axis=0)))
            else:
                answer.append([np.nan] * y_data.shape[1])
        return answer
