import unittest
import numpy as np
import pandas as pd
import pandas.testing as pdtest
# Module under test
from predictors.alternate_predictors import LastNPredictor, PerfectPredictor
# Modules needed for test setup
from misc.prediction_result import PredictionResult
from tests.utils_for_tests import mock_data, mock_data_last_n_predictor
from misc.dataset import StatsDataset
from misc.nn_helper_functions import stats_to_fantasy_points
import logging
import logging.config
from config.log_config import LOGGING_CONFIG

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')

class TestConstructor_PerfectPredictor(unittest.TestCase):
    # Set Up
    def setUp(self):
        pass

    def test_basic_attributes(self):
        name = 'test'
        predictor = PerfectPredictor(name)
        self.assertEqual(predictor.name, name)

    def test_input_name_of_different_type_works_fine(self):
        name = 29
        predictor = PerfectPredictor(name)
        self.assertEqual(predictor.name, name)
        
    # Tear Down
    def tearDown(self):
        pass

class TestEvalModel_PerfectPredictor(unittest.TestCase):
    def setUp(self):
        # Custom stats list (only using a subset of all statistics)
        self.scoring_weights = {'Pass Yds'   : 0.04,
                           'Rush Yds'   : 0.1,
                           'Rec Yds'    : 0.1
        }

        self.dataset = StatsDataset(name='dataset',
                                    pbp_df=mock_data.pbp_df,
                                    boxscore_df=mock_data.bs_df,
                                    id_df=mock_data.id_df)
        self.predictor = PerfectPredictor('test')
        self.truths = self.predictor.eval_truth(eval_data=self.dataset, scoring_weights=self.scoring_weights)

    def test_perfect_prediction_of_stats(self):
        pred_result = self.predictor.eval_model(self.dataset,
                                                scoring_weights=self.scoring_weights)
        pdtest.assert_frame_equal(pred_result.predicts,self.truths)        

    def test_generation_of_prediction_result(self):
        pred_result = self.predictor.eval_model(self.dataset,
                                                scoring_weights=self.scoring_weights)

        expected_pred_result = PredictionResult(dataset=self.dataset,
                                                 predicts=self.truths,
                                                 truths=self.truths, 
                                                 predictor_name=self.predictor.name,
                                                 scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(pred_result.predicts, expected_pred_result.predicts)
        pdtest.assert_frame_equal(pred_result.truths, expected_pred_result.truths)
        pdtest.assert_frame_equal(pred_result.pbp_df, expected_pred_result.pbp_df)
        pdtest.assert_frame_equal(pred_result.id_df, expected_pred_result.id_df)
        self.assertTrue(pred_result.dataset.equals(expected_pred_result.dataset))

    def test_improper_fantasy_point_kwargs_gives_error(self):
        self.scoring_weights['XYZ'] = 100
        with self.assertRaises(KeyError):
            self.predictor.eval_model(self.dataset, scoring_weights=self.scoring_weights)

    def test_input_normalized_false(self):
        truths_unnormalized = self.predictor.eval_truth(eval_data=self.dataset, normalized=False, scoring_weights=self.scoring_weights)
        pred_result = self.predictor.eval_model(self.dataset,
                                                normalized=False,
                                                scoring_weights=self.scoring_weights)
        expected_pred_result = PredictionResult(dataset=self.dataset,
                                                 predicts=truths_unnormalized,
                                                 truths=truths_unnormalized, 
                                                 normalized=False,
                                                 scoring_weights=self.scoring_weights)        

        pdtest.assert_frame_equal(pred_result.predicts, expected_pred_result.predicts)
        pdtest.assert_frame_equal(pred_result.truths, expected_pred_result.truths)
        pdtest.assert_frame_equal(pred_result.pbp_df, expected_pred_result.pbp_df)
        pdtest.assert_frame_equal(pred_result.id_df, expected_pred_result.id_df)
        self.assertTrue(pred_result.dataset.equals(expected_pred_result.dataset))

    def test_input_normalized_true(self):
        pred_result = self.predictor.eval_model(self.dataset,
                                                normalized=True,
                                                scoring_weights=self.scoring_weights)
        expected_pred_result = PredictionResult(dataset=self.dataset,
                                                 predicts=self.truths,
                                                 truths=self.truths, 
                                                 normalized=True,
                                                 scoring_weights=self.scoring_weights)        

        pdtest.assert_frame_equal(pred_result.predicts, expected_pred_result.predicts)
        pdtest.assert_frame_equal(pred_result.truths, expected_pred_result.truths)
        pdtest.assert_frame_equal(pred_result.pbp_df, expected_pred_result.pbp_df)
        pdtest.assert_frame_equal(pred_result.id_df, expected_pred_result.id_df)
        self.assertTrue(pred_result.dataset.equals(expected_pred_result.dataset))

    def tearDown(self):
        pass

class TestConstructor_LastNPredictor(unittest.TestCase):
    # Set Up
    def setUp(self):
        pass

    def test_basic_attributes(self):
        name = 'test'
        n = 3
        predictor = LastNPredictor(name,n)
        self.assertEqual(predictor.name, name)
        self.assertEqual(predictor.n, n)

    def test_no_input_n_defaults_to_one(self):
        name = 'test'
        predictor = LastNPredictor(name)
        self.assertEqual(predictor.n, 1)
        
    # Tear Down
    def tearDown(self):
        pass

class TestEvalModel_LastNPredictor(unittest.TestCase):
    def setUp(self):
        # Custom stats list (only using a subset of all statistics)
        self.scoring_weights = {'Pass Yds'   : 0.04,
                                'Rush Yds'   : 0.1,
                                'Rec Yds'    : 0.1
        }

        self.dataset = StatsDataset(name='dataset',
                                    pbp_df=mock_data_last_n_predictor.pbp_df,
                                    boxscore_df=mock_data_last_n_predictor.bs_df,
                                    id_df=mock_data_last_n_predictor.id_df)
        self.predictor = LastNPredictor('test',1)

    def test_last_one_game_score_returns_correct_result(self):
        result = self.predictor.eval_model(eval_data=self.dataset,
                                           all_data=self.dataset,
                                           scoring_weights=self.scoring_weights)

        # Get the last game's Fantasy Points by using the hard-coded map between a game and its previous
        stats = stats_to_fantasy_points(self.dataset.y_df, normalized=True, scoring_weights=self.scoring_weights)
        stats['Last Game Index'] = mock_data_last_n_predictor.map_to_last_game
        prev_game_score = stats.apply(lambda x: stats.loc[x['Last Game Index'],'Fantasy Points'] if x['Last Game Index']>=0 else 0, axis=1)

        pdtest.assert_series_equal(result.predicts['Fantasy Points'], prev_game_score, check_names=False)

    def test_last_two_games_score_returns_correct_result(self):
        self.predictor = LastNPredictor('test', 2)
        result = self.predictor.eval_model(eval_data=self.dataset,
                                           all_data=self.dataset,
                                           scoring_weights=self.scoring_weights)

        # Get the last game's Fantasy Points by using the hard-coded map between a game and its previous
        stats = stats_to_fantasy_points(self.dataset.y_df, normalized=True, scoring_weights=self.scoring_weights)
        stats['Last Game Index'] = mock_data_last_n_predictor.map_to_last_game
        prev_game_score = stats.apply(lambda x: stats.loc[x['Last Game Index'],'Fantasy Points'] if x['Last Game Index']>=0 else 0, axis=1)
        stats['Last Game Index'] = mock_data_last_n_predictor.map_to_second_to_last_game
        second_prev_game_score = stats.apply(lambda x: stats.loc[x['Last Game Index'],'Fantasy Points'] if x['Last Game Index']>=0 else np.nan, axis=1)
        avg_game_score = pd.concat([prev_game_score,second_prev_game_score],axis=1).mean(axis=1,skipna=True)

        pdtest.assert_series_equal(result.predicts['Fantasy Points'], avg_game_score, check_names=False)

    def test_missing_second_dataset_uses_first_dataset(self):
        result = self.predictor.eval_model(eval_data=self.dataset,
                                           scoring_weights=self.scoring_weights)

        # Get the last game's Fantasy Points by using the hard-coded map between a game and its previous
        stats = stats_to_fantasy_points(self.dataset.y_df, normalized=True, scoring_weights=self.scoring_weights)
        stats['Last Game Index'] = mock_data_last_n_predictor.map_to_last_game
        prev_game_score = stats.apply(lambda x: stats.loc[x['Last Game Index'],'Fantasy Points'] if x['Last Game Index']>=0 else 0, axis=1)

        pdtest.assert_series_equal(result.predicts['Fantasy Points'], prev_game_score, check_names=False)

    def test_improper_fantasy_point_kwargs_gives_error(self):
        self.scoring_weights['XYZ'] = 100
        with self.assertRaises(KeyError):
            self.predictor.eval_model(eval_data=self.dataset, all_data=self.dataset, scoring_weights=self.scoring_weights)

    def test_input_normalized_false(self):
        result = self.predictor.eval_model(self.dataset,
                                           normalized=False,
                                           scoring_weights=self.scoring_weights)

        # Get the last game's Fantasy Points by using the hard-coded map between a game and its previous
        stats = stats_to_fantasy_points(self.dataset.y_df, normalized=False, scoring_weights=self.scoring_weights)
        stats['Last Game Index'] = mock_data_last_n_predictor.map_to_last_game
        prev_game_score = stats.apply(lambda x: stats.loc[x['Last Game Index'],'Fantasy Points'] if x['Last Game Index']>=0 else 0, axis=1)

        pdtest.assert_series_equal(result.predicts['Fantasy Points'], prev_game_score, check_names=False)

    def test_input_normalized_true(self):
        result = self.predictor.eval_model(self.dataset,
                                           normalized=True,
                                           scoring_weights=self.scoring_weights)

        # Get the last game's Fantasy Points by using the hard-coded map between a game and its previous
        stats = stats_to_fantasy_points(self.dataset.y_df, normalized=True, scoring_weights=self.scoring_weights)
        stats['Last Game Index'] = mock_data_last_n_predictor.map_to_last_game
        prev_game_score = stats.apply(lambda x: stats.loc[x['Last Game Index'],'Fantasy Points'] if x['Last Game Index']>=0 else 0, axis=1)

        pdtest.assert_series_equal(result.predicts['Fantasy Points'], prev_game_score, check_names=False)

    def tearDown(self):
        pass
