import contextlib
import logging
import logging.config
import os
import unittest

import pandas.testing as pdtest
import yaml

from config.log_config import LOGGING_CONFIG
from misc.dataset import StatsDataset
from misc.stat_utils import stats_to_fantasy_points
from misc.yaml_constructor import add_yaml_constructors

# Module under test
from predictors import SleeperPredictor

# Modules needed for test setup
from tests._utils_for_tests import mock_data_predictors
from tests._utils_for_tests.skip_tests_config import SKIP_SLEEPER_TESTS

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")


# Data files config
test_data_files_config = "tests/_test_files/config_files/test_data_files_config.yaml"
add_yaml_constructors()


@unittest.skipIf(
    SKIP_SLEEPER_TESTS,
    "Sleeper API calls and file loads are slow. Skip flag can be set to False in skip_tests_config.py.",
)
class TestConstructor_SleeperPredictor(unittest.TestCase):
    # Set Up
    def setUp(self):
        # Sleeper data files
        with open(test_data_files_config) as file:
            self.data_files_config = yaml.safe_load(file)

    def test_basic_attributes_no_optional_inputs(self):
        name = "test"
        predictor = SleeperPredictor(name=name, data_files_config=self.data_files_config)

        self.assertEqual(predictor.name, name)
        self.assertEqual(predictor.player_id_file, self.data_files_config["master_player_id_file"])
        self.assertEqual(predictor.proj_dict_file, self.data_files_config["sleeper_proj_dict_file"])
        self.assertEqual(predictor.update_players, False)
        self.assertEqual(predictor.all_proj_dict, {})

    def test_basic_attributes_with_optional_inputs(self):
        name = "test"
        predictor = SleeperPredictor(
            name=name,
            data_files_config=self.data_files_config,
            update_players=False,
        )
        self.assertEqual(predictor.name, name)
        self.assertEqual(predictor.player_id_file, self.data_files_config["master_player_id_file"])
        self.assertEqual(predictor.proj_dict_file, self.data_files_config["sleeper_proj_dict_file"])
        self.assertEqual(predictor.update_players, False)
        self.assertEqual(predictor.all_proj_dict, {})

    def test_refresh_players_runs_without_error(self):
        # Cannot test correctness very easily...
        # Should provide a regression test before making any changes!
        SleeperPredictor(
            name="test",
            data_files_config=self.data_files_config,
            update_players=True,
        )

    # Tear Down
    def tearDown(self):
        pass


@unittest.skipIf(
    SKIP_SLEEPER_TESTS,
    "Sleeper API calls and file loads are slow. Skip flag can be set to False in skip_tests_config.py.",
)
class TestEvalModel_SleeperPredictor(unittest.TestCase):
    def setUp(self):
        # Sleeper data files
        with open(test_data_files_config) as file:
            self.data_files_config = yaml.safe_load(file)

        # Dummy non-existent files to use in a test
        self.nonexistent_file_1 = "tests/_test_files/empty/nonexistent_file2.json"
        self.nonexistent_file_2 = "tests/_test_files/empty/nonexistent_file2.json"
        # Dummy data files config pointing to non-existent file
        self.nonexistent_data_files_config = self.data_files_config.copy()
        self.nonexistent_data_files_config["master_player_id_file"] = self.nonexistent_file_1
        self.nonexistent_data_files_config["sleeper_proj_dict_file"] = self.nonexistent_file_2

        # Custom stats list (only using a subset of all statistics)
        # Custom dataset
        self.dataset = StatsDataset(
            name="dataset",
            id_df=mock_data_predictors.id_df,
            pbp_df=mock_data_predictors.pbp_df,
            boxscore_df=mock_data_predictors.bs_df,
            x_data_columns=mock_data_predictors.pbp_features,
            y_data_columns=mock_data_predictors.bs_features,
        )
        # Sleeper Predictor
        self.predictor = SleeperPredictor(
            name="test",
            data_files_config=self.data_files_config,
            update_players=False,
        )

    def test_eval_model_gives_correct_results(self):
        result = self.predictor.eval_model(eval_data=self.dataset)

        pdtest.assert_frame_equal(result.predicts, mock_data_predictors.expected_predicts_sleeper, check_dtype=False)

    def test_eval_model_nonexistent_data_files_gives_correct_results(self):
        self.predictor = SleeperPredictor(
            name="test",
            data_files_config=self.nonexistent_data_files_config,
        )
        result = self.predictor.eval_model(eval_data=self.dataset)

        pdtest.assert_frame_equal(result.predicts, mock_data_predictors.expected_predicts_sleeper, check_dtype=False)

    def test_year_outside_of_sleeper_data_gives_zeros(self):
        self.dataset.id_data["Year"] = (
            self.dataset.id_data["Year"] + 10
        )  # Set the year far in the future so that Sleeper can't find these games
        result = self.predictor.eval_model(eval_data=self.dataset)

        pdtest.assert_frame_equal(result.predicts, mock_data_predictors.expected_predicts_sleeper * 0, check_dtype=False)

    def test_player_outside_of_sleeper_data_gives_zeros(self):
        self.dataset.id_data["sleeper_id"] = self.dataset.id_data["sleeper_id"] * 10
        result = self.predictor.eval_model(eval_data=self.dataset)

        pdtest.assert_frame_equal(result.predicts, mock_data_predictors.expected_predicts_sleeper * 0, check_dtype=False)

    def test_input_normalized_false(self):
        result = self.predictor.eval_model(eval_data=self.dataset, normalized=False)

        pdtest.assert_frame_equal(result.predicts, mock_data_predictors.expected_predicts_sleeper, check_dtype=False)

    def test_input_normalized_true(self):
        result = self.predictor.eval_model(eval_data=self.dataset, normalized=True)
        expected_predicts_normalized_true = stats_to_fantasy_points(
            mock_data_predictors.expected_predicts_sleeper.drop("Fantasy Points", axis=1),
            stat_configs=self.dataset.y_data_columns,
            normalized=True,
        )

        pdtest.assert_frame_equal(result.predicts, expected_predicts_normalized_true, check_dtype=False)

    def tearDown(self):
        # Delete dummy nonexistent files
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.nonexistent_file_1)
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.nonexistent_file_2)
