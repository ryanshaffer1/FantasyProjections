import unittest
import pandas.testing as pdtest
import torch
from torch.utils.data import DataLoader
import os
import shutil
# Module under test
from predictors.neural_net_predictor import NeuralNetPredictor, NeuralNetwork
# Modules needed for test setup
from tests.utils_for_tests import mock_data_predictors
from misc.dataset import StatsDataset
from misc.nn_helper_functions import stats_to_fantasy_points
import logging
import logging.config
from config.log_config import LOGGING_CONFIG

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')

def compare_net_sizes(model1, model2):
    # Returns whether two Neural Networks are the same shape/size (i.e. same number of layers and parameters).
    # Also compares activation function types/layer types.
    # Does not compare whether the two models have the same values for their parameters, or names for their layers.

    for child1, child2 in zip(model1.children(), model2.children()):
        if (type(child1) != type(child2)) or (len(child1) != len(child2)):
            return False
        for layer1, layer2 in zip(child1, child2):
            if type(layer1) != type(layer2):
                return False
            if hasattr(layer1, 'in_features'):
                if not hasattr(layer2, 'in_features'):
                    return False
                if layer1.in_features != layer2.in_features:
                    return False
            if hasattr(layer1, 'out_features'):
                if not hasattr(layer2, 'out_features'):
                    return False
                if layer1.out_features != layer2.out_features:
                    return False                
    return True

class TestConstructor_NeuralNetPredictor(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.save_folder = 'data/test files/'
        self.load_folder = 'data/test files/'
        
        self.default_shape = {
            'players_input': 300,
            'teams_input': 32,
            'opps_input': 32,
            'stats_input': 25,
            'positions_input': 4,
            'embedding_player': 50,
            'embedding_team': 10,
            'embedding_opp': 10,
            'linear_stack': 300,
            'stats_output': 12,
        }
        self.shape_2 = {
            'players_input': 3,
            'teams_input': 2,
            'opps_input': 6,
            'stats_input': 5,
            'positions_input': 3,
            'embedding_player': 2,
            'embedding_team': 4,
            'embedding_opp': 4,
            'linear_stack': 30,
            'stats_output': 3,
        }
        
    def test_basic_attributes_no_optional_inputs(self):
        name = 'test'
        predictor = NeuralNetPredictor(name=name)
        
        self.assertEqual(predictor.name, name)
        self.assertIsNone(predictor.save_folder)
        self.assertIsNone(predictor.load_folder)
        self.assertIsNone(predictor.nn_shape)
        self.assertEqual(predictor.max_epochs, 100)
        self.assertEqual(predictor.n_epochs_to_stop, 5)
        self.assertTrue(predictor.device in ['cpu','mpu','cuda'])
        self.assertTrue(isinstance(predictor.model, NeuralNetwork))
        self.assertTrue(isinstance(predictor.optimizer,torch.optim.SGD))
        
    def test_basic_attributes_with_optional_inputs(self):
        name = 'test'
        predictor = NeuralNetPredictor(name=name,
                                       save_folder=self.save_folder,
                                       load_folder=self.load_folder,
                                       nn_shape=self.default_shape,
                                       max_epochs=1,
                                       n_epochs_to_stop=2)
        
        self.assertEqual(predictor.name, name)
        self.assertEqual(predictor.save_folder, self.save_folder)
        self.assertEqual(predictor.load_folder, self.load_folder)
        self.assertEqual(predictor.nn_shape, self.default_shape)
        self.assertEqual(predictor.max_epochs, 1)
        self.assertEqual(predictor.n_epochs_to_stop, 2)
        self.assertTrue(predictor.device in ['cpu','mpu','cuda'])
        self.assertTrue(isinstance(predictor.model, NeuralNetwork))
        self.assertTrue(isinstance(predictor.optimizer,torch.optim.SGD))

    def test_non_default_shape_gives_different_model(self):
        predictor1 = NeuralNetPredictor(name='test')
        predictor2 = NeuralNetPredictor(name='test_shape',
                                        nn_shape=self.shape_2)
        self.assertFalse(compare_net_sizes(predictor1.model, predictor2.model))

    # Tear Down
    def tearDown(self):
        pass

class TestLoadModel_NeuralNetPredictor(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.load_folder = 'data/test files/'
                
        self.default_shape = {
            'players_input': 300,
            'teams_input': 32,
            'opps_input': 32,
            'stats_input': 25,
            'positions_input': 4,
            'embedding_player': 50,
            'embedding_team': 10,
            'embedding_opp': 10,
            'linear_stack': 300,
            'stats_output': 12,
        }

    def test_loaded_model_equals_default(self):
        predictor = NeuralNetPredictor(name='test',
                                       load_folder=self.load_folder)
        expected_model = NeuralNetwork()

        self.assertTrue(compare_net_sizes(predictor.model, expected_model))

    def test_loaded_optimizer_equals_default(self):
        predictor = NeuralNetPredictor(name='test',
                                       load_folder=self.load_folder)
        expected_optimizer = torch.optim.SGD(predictor.model.parameters())
        
        self.assertEqual(predictor.optimizer.state_dict(), expected_optimizer.state_dict())
        
    def test_loaded_nn_shape_equals_default(self):
        predictor = NeuralNetPredictor(name='test',
                                       load_folder=self.load_folder)
        expected_shape = self.default_shape
        
        self.assertEqual(predictor.nn_shape, expected_shape)

    def test_invalid_folder_gives_error(self):
        with self.assertRaises(FileNotFoundError):
            NeuralNetPredictor(name='test',
                               load_folder='pizza')

    def test_empty_folder_gives_error(self):
        with self.assertRaises(FileNotFoundError):
            NeuralNetPredictor(name='test',
                               load_folder='data/test files/empty')

    def test_load_model_outside_of_initializer_overwrites_model(self):
        predictor = NeuralNetPredictor(name='test')
        # Change size of first linear layer in model
        list(predictor.model.children())[0][0].in_features = 4
        # Load model
        predictor.load(self.load_folder)
        
        self.assertNotEqual(list(predictor.model.children())[0][0].in_features, 4)

    # Tear Down
    def tearDown(self):
        pass

class TestSaveModel_NeuralNetPredictor(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.save_folder = 'data/test files/empty/'
        
    def test_save_model_can_be_reloaded(self):
        predictor = NeuralNetPredictor(name='test',
                                       save_folder=self.save_folder)
        predictor.save()

        predictor2 = NeuralNetPredictor(name='test',
                                        load_folder=self.save_folder)

        self.assertTrue(compare_net_sizes(predictor.model, predictor2.model))
        self.assertEqual(predictor.optimizer.state_dict(), predictor2.optimizer.state_dict())

    def test_additional_inputs_give_error(self):
        predictor = NeuralNetPredictor(name='test',
                                       save_folder=self.save_folder)
        with self.assertRaises(TypeError):
            predictor.save('filename')

    # Tear Down
    def tearDown(self):
        # Delete any files created in save folder
        for filename in os.listdir(self.save_folder):
            os.remove(self.save_folder+filename)

class TestEvalModel_NeuralNetPredictor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        
        self.save_folder = 'data/test files/'
        self.load_folder = 'data/test files/'
        self.mock_shape = {
            'players_input': 3,
            'teams_input': 2,
            'opps_input': 6,
            'stats_input': 5,
            'positions_input': 3,
            'embedding_player': 2,
            'embedding_team': 4,
            'embedding_opp': 4,
            'linear_stack': 30,
            'stats_output': 3,
        }

        # Custom stats list (only using a subset of all statistics)
        self.scoring_weights = {'Pass Yds'   : 0.04,
                                'Rush Yds'   : 0.1,
                                'Rec Yds'    : 0.1
        }
        # Custom dataset
        self.dataset = StatsDataset(name='dataset',
                                    pbp_df=mock_data_predictors.pbp_df_neural_net,
                                    boxscore_df=mock_data_predictors.bs_df,
                                    id_df=mock_data_predictors.id_df)
        # Custom dataset 2
        pbp_df_modified = mock_data_predictors.pbp_df_neural_net.copy()
        pbp_df_modified['Rush Yds'] *= 2
        self.dataset2 = StatsDataset(name='dataset',
                                     pbp_df=pbp_df_modified,
                                     boxscore_df=mock_data_predictors.bs_df,
                                     id_df=mock_data_predictors.id_df)

        # Neural Net Predictor
        self.predictor = NeuralNetPredictor(name='test',
                                            nn_shape=self.mock_shape)

    def test_eval_model_gives_correct_results(self):
        result = self.predictor.eval_model(eval_data=self.dataset,
                                           scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(result.predicts,mock_data_predictors.expected_predicts_neural_net, check_dtype=False)

    def test_different_input_dataset_gives_different_result(self):
        result = self.predictor.eval_model(eval_data=self.dataset2,
                                           scoring_weights=self.scoring_weights)
        with self.assertRaises(AssertionError):
            pdtest.assert_frame_equal(result.predicts,mock_data_predictors.expected_predicts_neural_net, check_dtype=False)

    def test_eval_model_with_dataloader_gives_correct_results(self):
        eval_dataloader = DataLoader(self.dataset, shuffle=False)
        result = self.predictor.eval_model(eval_dataloader=eval_dataloader,
                                           scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(result.predicts,mock_data_predictors.expected_predicts_neural_net, check_dtype=False)

    def test_eval_model_with_dataset_and_dataloader_favors_dataloader(self):
        eval_dataloader = DataLoader(self.dataset, shuffle=False)
        result = self.predictor.eval_model(eval_data=self.dataset2,
                                           eval_dataloader=eval_dataloader,
                                           scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(result.predicts,mock_data_predictors.expected_predicts_neural_net, check_dtype=False)

    def test_improper_fantasy_point_kwargs_gives_error(self):
        self.scoring_weights['XYZ'] = 100
        with self.assertRaises(KeyError):
            self.predictor.eval_model(eval_data=self.dataset, scoring_weights=self.scoring_weights)

    def test_input_normalized_false_raises_error(self):
        with self.assertRaises(ValueError):
            self.predictor.eval_model(eval_data=self.dataset,
                                      normalized=False,
                                      scoring_weights=self.scoring_weights)

    def test_input_normalized_true(self):
        result = self.predictor.eval_model(eval_data=self.dataset,
                                           normalized=True,
                                           scoring_weights=self.scoring_weights)

        pdtest.assert_frame_equal(result.predicts, mock_data_predictors.expected_predicts_neural_net, check_dtype=False)

    def tearDown(self):
        pass

class TestConfigureForTraining_NeuralNetPredictor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        
        self.save_folder = 'data/test files/'
        self.load_folder = 'data/test files/'
        self.mock_shape = {
            'players_input': 3,
            'teams_input': 2,
            'opps_input': 6,
            'stats_input': 5,
            'positions_input': 3,
            'embedding_player': 2,
            'embedding_team': 4,
            'embedding_opp': 4,
            'linear_stack': 30,
            'stats_output': 3,
        }

        # Custom stats list (only using a subset of all statistics)
        self.scoring_weights = {'Pass Yds'   : 0.04,
                                'Rush Yds'   : 0.1,
                                'Rec Yds'    : 0.1
        }
        # Custom dataset
        self.dataset = StatsDataset(name='dataset',
                                    pbp_df=mock_data_predictors.pbp_df_neural_net,
                                    boxscore_df=mock_data_predictors.bs_df,
                                    id_df=mock_data_predictors.id_df)
        # Custom dataset 2
        pbp_df_modified = mock_data_predictors.pbp_df_neural_net.copy()
        pbp_df_modified['Rush Yds'] *= 2
        self.dataset2 = StatsDataset(name='dataset',
                                     pbp_df=pbp_df_modified,
                                     boxscore_df=mock_data_predictors.bs_df,
                                     id_df=mock_data_predictors.id_df)

        # Neural Net Predictor
        self.predictor = NeuralNetPredictor(name='test',
                                            nn_shape=self.mock_shape)

    def test_stuff(self):
        pass

    def tearDown(self):
        pass
