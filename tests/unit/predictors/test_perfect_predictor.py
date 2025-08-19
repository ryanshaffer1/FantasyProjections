import logging
import logging.config
import unittest

import pandas.testing as pdtest

from config.log_config import LOGGING_CONFIG
from misc.dataset import StatsDataset

# Module under test
from predictors import PerfectPredictor

# Modules needed for test setup
from results import PredictionResult
from tests._utils_for_tests import mock_data

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")


class TestConstructor_PerfectPredictor(unittest.TestCase):
    # Set Up
    def setUp(self):
        pass

    def test_basic_attributes(self):
        name = "test"
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
        self.dataset = StatsDataset(
            name="dataset",
            id_df=mock_data.id_df,
            pbp_df=mock_data.pbp_df,
            boxscore_df=mock_data.bs_df,
            x_data_columns=mock_data.pbp_features,
            y_data_columns=mock_data.bs_features,
        )
        self.predictor = PerfectPredictor("test")
        self.truths = self.predictor.eval_truth(eval_data=self.dataset)

    def test_perfect_prediction_of_stats(self):
        pred_result = self.predictor.eval_model(self.dataset)
        pdtest.assert_frame_equal(pred_result.predicts, self.truths)

    def test_generation_of_prediction_result(self):
        pred_result = self.predictor.eval_model(self.dataset)

        expected_pred_result = PredictionResult(
            dataset=self.dataset,
            predicts=self.truths,
            truths=self.truths,
            predictor_name=self.predictor.name,
        )

        pdtest.assert_frame_equal(pred_result.predicts, expected_pred_result.predicts)
        pdtest.assert_frame_equal(pred_result.truths, expected_pred_result.truths)
        pdtest.assert_frame_equal(pred_result.pbp_df, expected_pred_result.pbp_df)
        pdtest.assert_frame_equal(pred_result.id_df, expected_pred_result.id_df)
        self.assertTrue(pred_result.dataset.equals(expected_pred_result.dataset))

    def test_input_normalized_false(self):
        truths_unnormalized = self.predictor.eval_truth(
            eval_data=self.dataset,
            normalized=False,
        )
        pred_result = self.predictor.eval_model(self.dataset, normalized=False)
        expected_pred_result = PredictionResult(
            dataset=self.dataset,
            predicts=truths_unnormalized,
            truths=truths_unnormalized,
            normalized=False,
        )

        pdtest.assert_frame_equal(pred_result.predicts, expected_pred_result.predicts)
        pdtest.assert_frame_equal(pred_result.truths, expected_pred_result.truths)
        pdtest.assert_frame_equal(pred_result.pbp_df, expected_pred_result.pbp_df)
        pdtest.assert_frame_equal(pred_result.id_df, expected_pred_result.id_df)
        self.assertTrue(pred_result.dataset.equals(expected_pred_result.dataset))

    def test_input_normalized_true(self):
        pred_result = self.predictor.eval_model(self.dataset, normalized=True)
        expected_pred_result = PredictionResult(
            dataset=self.dataset,
            predicts=self.truths,
            truths=self.truths,
            normalized=True,
        )

        pdtest.assert_frame_equal(pred_result.predicts, expected_pred_result.predicts)
        pdtest.assert_frame_equal(pred_result.truths, expected_pred_result.truths)
        pdtest.assert_frame_equal(pred_result.pbp_df, expected_pred_result.pbp_df)
        pdtest.assert_frame_equal(pred_result.id_df, expected_pred_result.id_df)
        self.assertTrue(pred_result.dataset.equals(expected_pred_result.dataset))

    def tearDown(self):
        pass
