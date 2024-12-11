import unittest
import pandas.testing as pdtest
# Module under test
from predictors import FantasyPredictor
# Modules needed for test setup
from results import PredictionResult
import tests.utils_for_tests.mock_data as mock_data
from misc.dataset import StatsDataset
from misc.stat_utils import stats_to_fantasy_points
import logging
import logging.config
from config.log_config import LOGGING_CONFIG

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')

class TestConstructor_FantasyPredictor(unittest.TestCase):
    # Set Up
    def setUp(self):
        pass

    def test_basic_attributes(self):
        name = 'test'
        predictor = FantasyPredictor(name)
        self.assertEqual(predictor.name, name)

    def test_input_of_different_type_works_fine(self):
        name = 29
        predictor = FantasyPredictor(name)
        self.assertEqual(predictor.name, name)
        
    # Tear Down
    def tearDown(self):
        pass

class TestEvalTruth_FantasyPredictor(unittest.TestCase):
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
        self.predictor = FantasyPredictor('test')
   
    def test_correct_calculation_of_fantasy_points(self):
        result = self.predictor.eval_truth(self.dataset, scoring_weights=self.scoring_weights)
        expected = stats_to_fantasy_points(self.dataset.y_df, normalized=True, scoring_weights=self.scoring_weights)
        pdtest.assert_frame_equal(result,expected)        

    def test_improper_fantasy_point_kwargs_gives_error(self):
        self.scoring_weights['XYZ'] = 100
        with self.assertRaises(KeyError):
            self.predictor.eval_truth(self.dataset, scoring_weights=self.scoring_weights)

    def test_input_normalized_false(self):
        result = self.predictor.eval_truth(self.dataset, normalized=False, scoring_weights=self.scoring_weights)
        expected = stats_to_fantasy_points(self.dataset.y_df, normalized=False, scoring_weights=self.scoring_weights)
        pdtest.assert_frame_equal(result,expected)        

    def tearDown(self):
        pass

class TestGenPredictionResult_FantasyPredictor(unittest.TestCase):
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
        self.predictor = FantasyPredictor('test')
        self.predicts = self.predictor.eval_truth(eval_data=self.dataset, scoring_weights=self.scoring_weights)
        self.truths = self.predictor.eval_truth(eval_data=self.dataset, scoring_weights=self.scoring_weights)
   
    def test_generation_of_prediction_result(self):
        pred_result = self.predictor._gen_prediction_result(self.predicts,
                                                            self.truths,
                                                            self.dataset,
                                                            scoring_weights=self.scoring_weights)

        expected_pred_result = PredictionResult(dataset=self.dataset,
                                                 predicts=self.predicts,
                                                 truths=self.truths, 
                                                 predictor_name=self.predictor.name,
                                                 scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(pred_result.predicts, expected_pred_result.predicts)
        pdtest.assert_frame_equal(pred_result.truths, expected_pred_result.truths)
        pdtest.assert_frame_equal(pred_result.pbp_df, expected_pred_result.pbp_df)
        pdtest.assert_frame_equal(pred_result.id_df, expected_pred_result.id_df)
        self.assertTrue(pred_result.dataset.equals(expected_pred_result.dataset))

    def tearDown(self):
        pass
