import unittest
import pandas.testing as pdtest
# Module under test
from results import PredictionResult, PredictionResultGroup
# Modules needed for test setup
import tests.utils_for_tests.mock_data as mock_data
from misc.dataset import StatsDataset
from misc.stat_utils import stats_to_fantasy_points
from predictors.perfect_predictor import PerfectPredictor
import logging
import logging.config
from config.log_config import LOGGING_CONFIG

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')

class TestConstructor_PredictionResult(unittest.TestCase):
    # Set Up
    def setUp(self):
        # Custom stats list (only using a subset of all statistics)
        self.scoring_weights = {'Pass Yds'   : 0.04,
                           'Rush Yds'   : 0.1,
                           'Rec Yds'    : 0.1
        }

        self.dataset = StatsDataset(name='dataset',
                                    id_df=mock_data.id_df,
                                    pbp_df=mock_data.pbp_df,
                                    boxscore_df=mock_data.bs_df)

        # Create a Predictor: perfect prediction model
        self.predictor = PerfectPredictor(name='Perfect Predictor')
        # Predicted and true stats
        self.predicts = self.predictor.eval_truth(eval_data=self.dataset, scoring_weights=self.scoring_weights)
        self.truths = self.predictor.eval_truth(eval_data=self.dataset, scoring_weights=self.scoring_weights)

    def test_basic_attributes(self):
        prediction_result = PredictionResult(dataset=self.dataset,
                                             predicts=self.predicts,
                                             truths=self.truths, 
                                             predictor_name=self.predictor.name,
                                             scoring_weights=self.scoring_weights)
        
        pdtest.assert_frame_equal(prediction_result.predicts, self.predicts)
        pdtest.assert_frame_equal(prediction_result.truths, self.truths)
        self.assertTrue(prediction_result.dataset.equals(self.dataset))
        self.assertEqual(prediction_result.predictor_name,self.predictor.name)
        pdtest.assert_frame_equal(prediction_result.id_df,self.dataset.id_data.reset_index(drop=True))
    
    def test_pbp_with_fantasy_points(self):
        prediction_result = PredictionResult(dataset=self.dataset,
                                             predicts=self.predicts,
                                             truths=self.truths, 
                                             predictor_name=self.predictor.name,
                                             scoring_weights=self.scoring_weights)
        expected_pbp_result = stats_to_fantasy_points(self.dataset.x_data, stat_indices=self.dataset.x_data_columns,
                                                      normalized=True, scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(prediction_result.pbp_df,expected_pbp_result)
    
    def test_pbp_with_fantasy_points_normalized_false(self):
        prediction_result = PredictionResult(dataset=self.dataset,
                                             predicts=self.predicts,
                                             truths=self.truths, 
                                             predictor_name=self.predictor.name,
                                             normalized=False,
                                             scoring_weights=self.scoring_weights)
        expected_pbp_result = stats_to_fantasy_points(self.dataset.x_data, stat_indices=self.dataset.x_data_columns,
                                                      normalized=False, scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(prediction_result.pbp_df,expected_pbp_result)
    
    def test_missing_stats_gives_error(self):
        with self.assertRaises(TypeError):
            PredictionResult(dataset=self.dataset,
                             truths=self.truths,
                             predictor_name=self.predictor.name,
                             scoring_weights=self.scoring_weights)
        with self.assertRaises(TypeError):
            PredictionResult(dataset=self.dataset,
                             predicts=self.predicts,
                             predictor_name=self.predictor.name,
                             scoring_weights=self.scoring_weights)

    def test_missing_dataset_gives_error(self):
        with self.assertRaises(TypeError):
            PredictionResult(predicts=self.predicts,
                             truths=self.truths,
                             predictor_name=self.predictor.name,
                             scoring_weights=self.scoring_weights)
    
    def test_invalid_dataset_type_gives_error(self):
        with self.assertRaises(AttributeError):
            PredictionResult(dataset=self.dataset.id_data,
                             predicts=self.predicts,
                             truths=self.truths,
                             predictor_name=self.predictor.name,
                             scoring_weights=self.scoring_weights)

    def test_optional_predictor_name_not_passed_defaults_to_unknown(self):
        prediction_result = PredictionResult(dataset=self.dataset,
                                             predicts=self.predicts,
                                             truths=self.truths, 
                                             scoring_weights=self.scoring_weights)
        self.assertEqual(prediction_result.predictor_name,'Unknown Predictor')
    
    # Tear Down
    def tearDown(self):
        pass

class TestDiffPredVsTruth_PredictionResult(unittest.TestCase):
    # Set Up
    def setUp(self):
        # Custom stats list (only using a subset of all statistics)
        self.scoring_weights = {'Pass Yds'   : 0.04,
                           'Rush Yds'   : 0.1,
                           'Rec Yds'    : 0.1
        }

        self.dataset = StatsDataset(name='dataset',
                                    id_df=mock_data.id_df,
                                    pbp_df=mock_data.pbp_df,
                                    boxscore_df=mock_data.bs_df)

        # Create a Predictor: perfect prediction model
        self.predictor = PerfectPredictor(name='Perfect Predictor')
        # Predicted and true stats
        self.predicts = self.predictor.eval_truth(eval_data=self.dataset, scoring_weights=self.scoring_weights)
        self.truths = self.predictor.eval_truth(eval_data=self.dataset, scoring_weights=self.scoring_weights)
        # Prediction Result
        self.prediction_result = PredictionResult(dataset=self.dataset,
                                                 predicts=self.predicts,
                                                 truths=self.truths, 
                                                 predictor_name=self.predictor.name,
                                                 scoring_weights=self.scoring_weights)
        # Compute average difference between prediction and truth
        self.abs_diff = abs(self.predicts['Fantasy Points'] - self.truths['Fantasy Points']).values.tolist()
        self.signed_diff = (self.predicts['Fantasy Points'] - self.truths['Fantasy Points']).values.tolist()

    def test_absolute_error(self):
        result = self.prediction_result.diff_pred_vs_truth(absolute=True)
        self.assertEqual(result,self.abs_diff)
    
    def test_signed_error(self):
        result = self.prediction_result.diff_pred_vs_truth(absolute=False)
        self.assertEqual(result,self.signed_diff)
    
    def test_no_input_defaults_to_signed_error(self):
        result = self.prediction_result.diff_pred_vs_truth()
        self.assertEqual(result,self.signed_diff)
    
    def test_invalid_input_gives_error(self):
        result = self.prediction_result.diff_pred_vs_truth(absolute=39)
        self.assertEqual(result,self.signed_diff)
    
    # Tear Down
    def tearDown(self):
        pass

class TestConstructor_PredictionResultGroup(unittest.TestCase):
    # Set Up
    def setUp(self):
        # Custom stats list (only using a subset of all statistics)
        self.scoring_weights = {'Pass Yds'   : 0.04,
                           'Rush Yds'   : 0.1,
                           'Rec Yds'    : 0.1
        }

        self.dataset = StatsDataset(name='dataset',
                                    id_df=mock_data.id_df,
                                    pbp_df=mock_data.pbp_df,
                                    boxscore_df=mock_data.bs_df)

        # Create two Predictors: perfect prediction models
        self.predictor1 = PerfectPredictor(name='Perfect Predictor 1')
        self.predictor2 = PerfectPredictor(name='Perfect Predictor 2')
        # Predicted and true stats
        self.predicts = self.predictor1.eval_truth(eval_data=self.dataset, scoring_weights=self.scoring_weights)
        self.truths = self.predictor1.eval_truth(eval_data=self.dataset, scoring_weights=self.scoring_weights)
        # PredictionResult objects
        self.prediction_result1 = PredictionResult(dataset=self.dataset,
                                                   predicts=self.predicts,
                                                   truths=self.truths, 
                                                   predictor_name=self.predictor1.name,
                                                   scoring_weights=self.scoring_weights)
        self.prediction_result2 = PredictionResult(dataset=self.dataset,
                                                   predicts=self.predicts,
                                                   truths=self.truths, 
                                                   predictor_name=self.predictor2.name,
                                                   scoring_weights=self.scoring_weights)

    def test_basic_attributes(self):
        group = PredictionResultGroup([self.prediction_result1, self.prediction_result2])
        self.assertEqual(group.results,[self.prediction_result1, self.prediction_result2])
        self.assertEqual(group.names,['Perfect Predictor 1', 'Perfect Predictor 2'])

    def test_tuple_input_works_as_expected(self):
        tuple_result = PredictionResultGroup((self.prediction_result1,self.prediction_result2))
        self.assertEqual(tuple_result.results,(self.prediction_result1, self.prediction_result2))

    def test_singular_input_works_as_expected(self):
        group = PredictionResultGroup([self.prediction_result1])
        self.assertEqual(group.results,[self.prediction_result1])

    def test_non_predictionresult_inputs_gives_error(self):
        with self.assertRaises(TypeError):
            PredictionResultGroup(self.prediction_result1)

    # Tear Down
    def tearDown(self):
        pass
