import logging
import logging.config
import unittest

import numpy as np
import pandas as pd
import pandas.testing as pdtest

from config.log_config import LOGGING_CONFIG
from misc.dataset import StatsDataset
from misc.stat_utils import stats_to_fantasy_points

# Module under test
from predictors import LastNPredictor

# Modules needed for test setup
from tests._utils_for_tests import mock_data_predictors

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")


class TestConstructor_LastNPredictor(unittest.TestCase):
    # Set Up
    def setUp(self):
        pass

    def test_basic_attributes(self):
        name = "test"
        n = 3
        predictor = LastNPredictor(name,n)
        self.assertEqual(predictor.name, name)
        self.assertEqual(predictor.n, n)

    def test_no_input_n_defaults_to_one(self):
        name = "test"
        predictor = LastNPredictor(name)
        self.assertEqual(predictor.n, 1)

    # Tear Down
    def tearDown(self):
        pass

class TestEvalModel_LastNPredictor(unittest.TestCase):
    def setUp(self):
        # Custom stats list (only using a subset of all statistics)
        self.scoring_weights = {"Pass Yds"   : 0.04,
                                "Rush Yds"   : 0.1,
                                "Rec Yds"    : 0.1,
        }

        self.dataset = StatsDataset(name="dataset",
                                    id_df=mock_data_predictors.id_df,
                                    pbp_df=mock_data_predictors.pbp_df,
                                    boxscore_df=mock_data_predictors.bs_df)
        self.predictor = LastNPredictor("test",1)

    def test_last_one_game_score_returns_correct_result(self):
        result = self.predictor.eval_model(eval_data=self.dataset,
                                           all_data=self.dataset,
                                           scoring_weights=self.scoring_weights)

        # Get the last game's Fantasy Points by using the hard-coded map between a game and its previous
        stats = stats_to_fantasy_points(self.dataset.y_data, stat_indices=self.dataset.y_data_columns,
                                        normalized=True, scoring_weights=self.scoring_weights)
        stats["Last Game Index"] = mock_data_predictors.map_to_last_game
        prev_game_score = stats.apply(lambda x: stats.loc[x["Last Game Index"],"Fantasy Points"] if x["Last Game Index"]>=0 else 0, axis=1)

        pdtest.assert_series_equal(result.predicts["Fantasy Points"], prev_game_score, check_names=False)

    def test_last_two_games_score_returns_correct_result(self):
        self.predictor = LastNPredictor("test", 2)
        result = self.predictor.eval_model(eval_data=self.dataset,
                                           all_data=self.dataset,
                                           scoring_weights=self.scoring_weights)

        # Get the last game's Fantasy Points by using the hard-coded map between a game and its previous
        stats = stats_to_fantasy_points(self.dataset.y_data, stat_indices=self.dataset.y_data_columns,
                                        normalized=True, scoring_weights=self.scoring_weights)
        stats["Last Game Index"] = mock_data_predictors.map_to_last_game
        prev_game_score = stats.apply(lambda x: stats.loc[x["Last Game Index"],"Fantasy Points"] if x["Last Game Index"]>=0 else 0, axis=1)
        stats["Last Game Index"] = mock_data_predictors.map_to_second_to_last_game
        second_prev_game_score = stats.apply(lambda x: stats.loc[x["Last Game Index"],"Fantasy Points"] if x["Last Game Index"]>=0 else np.nan, axis=1)
        avg_game_score = pd.concat([prev_game_score,second_prev_game_score],axis=1).mean(axis=1,skipna=True)

        pdtest.assert_series_equal(result.predicts["Fantasy Points"], avg_game_score, check_names=False)

    def test_missing_second_dataset_uses_first_dataset(self):
        result = self.predictor.eval_model(eval_data=self.dataset,
                                           scoring_weights=self.scoring_weights)

        # Get the last game's Fantasy Points by using the hard-coded map between a game and its previous
        stats = stats_to_fantasy_points(self.dataset.y_data, stat_indices=self.dataset.y_data_columns,
                                        normalized=True, scoring_weights=self.scoring_weights)
        stats["Last Game Index"] = mock_data_predictors.map_to_last_game
        prev_game_score = stats.apply(lambda x: stats.loc[x["Last Game Index"],"Fantasy Points"] if x["Last Game Index"]>=0 else 0, axis=1)

        pdtest.assert_series_equal(result.predicts["Fantasy Points"], prev_game_score, check_names=False)

    def test_improper_fantasy_point_kwargs_gives_error(self):
        self.scoring_weights["XYZ"] = 100
        with self.assertRaises(KeyError):
            self.predictor.eval_model(eval_data=self.dataset, all_data=self.dataset, scoring_weights=self.scoring_weights)

    def test_input_normalized_false(self):
        result = self.predictor.eval_model(self.dataset,
                                           normalized=False,
                                           scoring_weights=self.scoring_weights)

        # Get the last game's Fantasy Points by using the hard-coded map between a game and its previous
        stats = stats_to_fantasy_points(self.dataset.y_data, stat_indices=self.dataset.y_data_columns,
                                        normalized=False, scoring_weights=self.scoring_weights)
        stats["Last Game Index"] = mock_data_predictors.map_to_last_game
        prev_game_score = stats.apply(lambda x: stats.loc[x["Last Game Index"],"Fantasy Points"] if x["Last Game Index"]>=0 else 0, axis=1)

        pdtest.assert_series_equal(result.predicts["Fantasy Points"], prev_game_score, check_names=False)

    def test_input_normalized_true(self):
        result = self.predictor.eval_model(self.dataset,
                                           normalized=True,
                                           scoring_weights=self.scoring_weights)

        # Get the last game's Fantasy Points by using the hard-coded map between a game and its previous
        stats = stats_to_fantasy_points(self.dataset.y_data, stat_indices=self.dataset.y_data_columns,
                                        normalized=True, scoring_weights=self.scoring_weights)
        stats["Last Game Index"] = mock_data_predictors.map_to_last_game
        prev_game_score = stats.apply(lambda x: stats.loc[x["Last Game Index"],"Fantasy Points"] if x["Last Game Index"]>=0 else 0, axis=1)

        pdtest.assert_series_equal(result.predicts["Fantasy Points"], prev_game_score, check_names=False)

    def tearDown(self):
        pass
