import unittest
import pandas.testing as pdtest
import os
import shutil
# Module under test
from predictors import SleeperPredictor
# Modules needed for test setup
from tests.utils_for_tests import mock_data_predictors
from tests.utils_for_tests.skip_tests_config import SKIP_SLEEPER_TESTS
from misc.dataset import StatsDataset
from misc.stat_utils import stats_to_fantasy_points
import logging
import logging.config
from config.log_config import LOGGING_CONFIG

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')

@unittest.skipIf(SKIP_SLEEPER_TESTS,'Sleeper API calls and file loads are slow. Skip flag can be set to False in skip_tests_config.py.')
class TestConstructor_SleeperPredictor(unittest.TestCase):
    # Set Up
    def setUp(self):
        # Sleeper data files
        self.sleeper_player_dict_file = 'data/misc/sleeper_player_dict.json'
        self.sleeper_proj_dict_file = 'data/misc/sleeper_projections_dict.json'
        # Make a copy of the player dict (it gets overwritten during a test)
        self.player_dict_copy_file = 'data/misc/sleeper_player_dict_copy.json'
        shutil.copyfile(self.sleeper_player_dict_file, self.player_dict_copy_file)
        
    def test_basic_attributes_no_optional_inputs(self):
        name = 'test'
        try:
            predictor = SleeperPredictor(name=name)
        except Exception as e:
            self.fail(f'SleeperPredictor init without data files raised {type(e)} unexpectedly.')
        
        self.assertEqual(predictor.name, name)
        self.assertIsNone(predictor.player_dict_file)
        self.assertIsNone(predictor.proj_dict_file)
        self.assertEqual(predictor.update_players, True) # Default value of False is overwritten because no player data was passed in
        self.assertEqual(predictor.all_proj_dict, {})

    def test_basic_attributes_with_optional_inputs(self):
        name = 'test'
        predictor = SleeperPredictor(name=name,
                                     player_dict_file=self.player_dict_copy_file,
                                     proj_dict_file=self.sleeper_proj_dict_file,
                                     update_players=False)
        self.assertEqual(predictor.name, name)
        self.assertEqual(predictor.player_dict_file, self.player_dict_copy_file)
        self.assertEqual(predictor.proj_dict_file, self.sleeper_proj_dict_file)
        self.assertEqual(predictor.update_players, False)
        self.assertEqual(predictor.all_proj_dict, {})

    def test_refresh_players_runs_without_error(self):
        # Cannot test correctness very easily...
        # Should provide a regression test before making any changes!
        try:
            SleeperPredictor(name='test',
                             player_dict_file=self.player_dict_copy_file,
                             proj_dict_file=self.sleeper_proj_dict_file,
                             update_players=True)
        except Exception as e:
            self.fail(f'SleeperPredictor.refresh_players raised {type(e)} unexpectedly.')

    # Tear Down
    def tearDown(self):
        # Delete copy of player dictionary
        os.remove(self.player_dict_copy_file)

@unittest.skipIf(SKIP_SLEEPER_TESTS,'Sleeper API calls and file loads are slow. Skip flag can be set to False in skip_tests_config.py.')
class TestEvalModel_SleeperPredictor(unittest.TestCase):
    def setUp(self):
        # Sleeper data files
        self.sleeper_player_dict_file = 'data/misc/sleeper_player_dict.json'
        self.sleeper_proj_dict_file = 'data/misc/sleeper_projections_dict.json'
        # Make a copy of the projections dict (it gets overwritten during a test)
        self.proj_dict_copy_file = 'data/misc/sleeper_projections_dict_copy.json'
        shutil.copyfile(self.sleeper_proj_dict_file, self.proj_dict_copy_file)
        # Dummy non-existent files to use in a test
        self.nonexistent_file_1 = 'data/misc/nonexistent_file1.json'
        self.nonexistent_file_2 = 'data/misc/nonexistent_file2.json'

        # Custom stats list (only using a subset of all statistics)
        self.scoring_weights = {'Pass Yds'   : 0.04,
                                'Rush Yds'   : 0.1,
                                'Rec Yds'    : 0.1
        }
        # Custom dataset
        self.dataset = StatsDataset(name='dataset',
                                    id_df=mock_data_predictors.id_df,
                                    pbp_df=mock_data_predictors.pbp_df,
                                    boxscore_df=mock_data_predictors.bs_df)
        # Sleeper Predictor
        self.predictor = SleeperPredictor(name='test',
                                          player_dict_file=self.sleeper_player_dict_file,
                                          proj_dict_file=self.proj_dict_copy_file,
                                          update_players=False)

    def test_eval_model_gives_correct_results(self):
        result = self.predictor.eval_model(eval_data=self.dataset, scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(result.predicts,mock_data_predictors.expected_predicts_sleeper, check_dtype=False)

    def test_eval_model_no_data_input_gives_correct_results(self):
        self.predictor = SleeperPredictor(name='test')
        result = self.predictor.eval_model(eval_data=self.dataset, scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(result.predicts,mock_data_predictors.expected_predicts_sleeper, check_dtype=False)

    def test_eval_model_nonexistent_data_files_gives_correct_results(self):
        self.predictor = SleeperPredictor(name='test',
                                          player_dict_file=self.nonexistent_file_1,
                                          proj_dict_file=self.nonexistent_file_2)
        result = self.predictor.eval_model(eval_data=self.dataset, scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(result.predicts,mock_data_predictors.expected_predicts_sleeper, check_dtype=False)

    def test_year_outside_of_sleeper_data_gives_zeros(self):
        self.dataset.id_data['Year'] = self.dataset.id_data['Year']+10 # Set the year far in the future so that Sleeper can't find these games
        result = self.predictor.eval_model(eval_data=self.dataset, scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(result.predicts, mock_data_predictors.expected_predicts_sleeper*0, check_dtype=False)

    def test_player_outside_of_sleeper_data_gives_zeros(self):
        self.dataset.id_data['Player'] = self.dataset.id_data['Player'] + '_test'
        result = self.predictor.eval_model(eval_data=self.dataset, scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(result.predicts, mock_data_predictors.expected_predicts_sleeper*0, check_dtype=False)

    def test_improper_fantasy_point_kwargs_gives_error(self):
        self.scoring_weights['XYZ'] = 100
        with self.assertRaises(KeyError):
            self.predictor.eval_model(eval_data=self.dataset, scoring_weights=self.scoring_weights)

    def test_input_normalized_false(self):
        result = self.predictor.eval_model(eval_data=self.dataset,
                                           normalized=False,
                                           scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(result.predicts,mock_data_predictors.expected_predicts_sleeper, check_dtype=False)

    def test_input_normalized_true(self):
        result = self.predictor.eval_model(eval_data=self.dataset,
                                           normalized=True,
                                           scoring_weights=self.scoring_weights)
        expected_predicts_normalized_true = stats_to_fantasy_points(mock_data_predictors.expected_predicts_sleeper,
                                                                    normalized=True,
                                                                    scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(result.predicts,expected_predicts_normalized_true, check_dtype=False)

    def tearDown(self):
        # Delete copy of projections dictionary
        os.remove(self.proj_dict_copy_file)
        # Delete dummy nonexistent files
        try:
            os.remove(self.nonexistent_file_1)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.nonexistent_file_2)
        except FileNotFoundError:
            pass
