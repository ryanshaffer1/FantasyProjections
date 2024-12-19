import unittest
import pandas.testing as pdtest
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
# Module under test
from predictors import NeuralNetPredictor
# Modules needed for test setup
from neural_net import NeuralNetwork, HyperParameter, HyperParameterSet
from neural_net.nn_utils import compare_net_sizes
from tests.utils_for_tests import mock_data_predictors
from config.hp_config import hp_defaults
from config.nn_config import default_nn_shape, nn_train_settings
from misc.dataset import StatsDataset
import logging
import logging.config
from config.log_config import LOGGING_CONFIG

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')

def check_all_model_layers_changed(model1, model2):
    # Returns whether two Neural Networks have different parameters in every layer of the network.
    # Used to verify that the training process affects every layer of the network (and there are no "isolated" layers).

    if not compare_net_sizes(model1,model2):
        raise ValueError('Models are not the same size/shape')

    for child1, child2 in zip(model1.children(), model2.children()):
        for layer1, layer2 in zip(child1, child2):
            if hasattr(layer1, '_parameters'):
                for tensor1, tensor2 in zip(layer1._parameters.values(), layer2._parameters.values()):
                    if torch.equal(tensor1,tensor2):
                        return False
    return True


class TestConstructor_NeuralNetPredictor(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.save_folder = 'data/test files/'
        self.load_folder = 'data/test files/'

        self.shape_2 = {
            'players_input': 3,
            'teams_input': 2,
            'opps_input': 6,
            'stats_input': 8,
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
        self.assertEqual(predictor.nn_shape, default_nn_shape)
        self.assertEqual(predictor.max_epochs, nn_train_settings['max_epochs'])
        self.assertEqual(predictor.n_epochs_to_stop, nn_train_settings['n_epochs_to_stop'])
        self.assertTrue(predictor.device in ['cpu','mpu','cuda'])
        self.assertTrue(isinstance(predictor.model, NeuralNetwork))
        self.assertTrue(isinstance(predictor.optimizer,torch.optim.SGD))
        
    def test_basic_attributes_with_optional_inputs(self):
        name = 'test'
        predictor = NeuralNetPredictor(name=name,
                                       save_folder=self.save_folder,
                                       load_folder=self.load_folder,
                                       nn_shape=default_nn_shape,
                                       max_epochs=1,
                                       n_epochs_to_stop=2)
        
        self.assertEqual(predictor.name, name)
        self.assertEqual(predictor.save_folder, self.save_folder)
        self.assertEqual(predictor.load_folder, self.load_folder)
        self.assertEqual(predictor.nn_shape, default_nn_shape)
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

        self.weird_shape = {
            'players_input': 1,
            'teams_input': 1,
            'opps_input': 1,
            'stats_input': 1,
            'embedding_player': 1,
            'embedding_team': 1,
            'embedding_opp': 1,
            'linear_stack': 1,
            'stats_output': 1,
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
        
        self.assertEqual(predictor.nn_shape, default_nn_shape)

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

    def test_load_model_of_new_shape_overwrites_model(self):
        predictor = NeuralNetPredictor(name='test',
                                       nn_shape = self.weird_shape)
        predictor.load(self.load_folder)
        expected_model = NeuralNetPredictor(load_folder=self.load_folder).model

        self.assertTrue(compare_net_sizes(predictor.model, expected_model))

    def test_load_model_of_different_shape_than_input_shape_overwrites_input_shape(self):
        predictor = NeuralNetPredictor(name='test',
                                       nn_shape = self.weird_shape,
                                       load_folder=self.load_folder)
        expected_pred = NeuralNetPredictor(load_folder=self.load_folder)

        self.assertTrue(compare_net_sizes(predictor.model, expected_pred.model))
        self.assertEqual(predictor.nn_shape, expected_pred.nn_shape)

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
            'stats_input': 8,
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
                                    id_df=mock_data_predictors.id_df,
                                    pbp_df=mock_data_predictors.pbp_df_neural_net,
                                    boxscore_df=mock_data_predictors.bs_df)
        # Custom dataset 2
        pbp_df_modified = mock_data_predictors.pbp_df_neural_net.copy()
        pbp_df_modified['Rush Yds'] *= 2
        self.dataset2 = StatsDataset(name='dataset',
                                     id_df=mock_data_predictors.id_df,
                                     pbp_df=pbp_df_modified,
                                     boxscore_df=mock_data_predictors.bs_df)

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

class TestModifyHyperParameterValues_NeuralNetPredictor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mock_shape = {
            'players_input': 3,
            'teams_input': 2,
            'opps_input': 6,
            'stats_input': 8,
            'embedding_player': 2,
            'embedding_team': 4,
            'embedding_opp': 4,
            'linear_stack': 30,
            'stats_output': 3,
        }

        # Hyper-parameters
        self.mini_batch_size = 100
        self.learning_rate = 1
        self.lmbda = 0.001
        self.loss_fn = nn.CrossEntropyLoss()
        self.linear_stack = 200
        self.mini_batch_size_hp = HyperParameter('mini_batch_size', optimizable=False, 
                                                 value=self.mini_batch_size)
        self.learning_rate_hp = HyperParameter('learning_rate', optimizable=False,
                                               value=self.learning_rate)
        self.lmbda_hp = HyperParameter('lmbda', optimizable=False,
                                       value=self.lmbda)
        self.loss_fn_hp = HyperParameter('loss_fn', optimizable=False,
                                        value=self.loss_fn)
        self.linear_stack_hp = HyperParameter('linear_stack',optimizable=False,
                                              value=self.linear_stack)
        self.hp_set = HyperParameterSet((self.mini_batch_size_hp,self.learning_rate_hp,self.lmbda_hp,self.loss_fn_hp,self.linear_stack_hp),optimize=False)
        self.hp_dict = {'mini_batch_size': int(self.mini_batch_size),
                        'learning_rate': self.learning_rate,
                        'lmbda': self.lmbda,
                        'loss_fn': self.loss_fn,
                        'linear_stack': self.linear_stack}

        # Custom stats list (only using a subset of all statistics)
        self.scoring_weights = {'Pass Yds'   : 0.04,
                                'Rush Yds'   : 0.1,
                                'Rec Yds'    : 0.1
        }

        # Neural Net Predictor
        self.predictor = NeuralNetPredictor(name='test',
                                            nn_shape=self.mock_shape)

    def test_hyper_parameter_set_with_all_hps_configures_all_correctly(self):
        self.predictor.modify_hyper_parameter_values(param_set=self.hp_set)
        
        self.assertEqual(self.predictor.mini_batch_size, self.hp_set.get('mini_batch_size').value)
        self.assertEqual(self.predictor.learning_rate, self.hp_set.get('learning_rate').value)
        self.assertEqual(self.predictor.lmbda, self.hp_set.get('lmbda').value)
        self.assertEqual(self.predictor.loss_fn, self.hp_set.get('loss_fn').value)
        self.assertEqual(self.predictor.nn_shape['linear_stack'], self.hp_set.get('linear_stack').value)
    
    def test_hyper_parameter_set_with_some_hps_configures_all_correctly(self):
        partial_hp_set = HyperParameterSet((self.mini_batch_size_hp, self.learning_rate_hp),optimize=False)
        self.predictor.modify_hyper_parameter_values(param_set=partial_hp_set)
        
        self.assertEqual(self.predictor.mini_batch_size, self.hp_set.get('mini_batch_size').value)
        self.assertEqual(self.predictor.learning_rate, self.hp_set.get('learning_rate').value)
        self.assertEqual(self.predictor.lmbda, hp_defaults['lmbda']['value'])
        self.assertEqual(self.predictor.loss_fn, hp_defaults['loss_fn']['value'])
        self.assertEqual(self.predictor.nn_shape['linear_stack'], self.mock_shape['linear_stack'])
    
    def test_empty_hyper_parameter_set_configures_all_correctly(self):
        empty_hp_set = HyperParameterSet([],optimize=False)
        self.predictor.modify_hyper_parameter_values(param_set=empty_hp_set)
        
        self.assertEqual(self.predictor.mini_batch_size, hp_defaults['mini_batch_size']['value'])
        self.assertEqual(self.predictor.learning_rate, hp_defaults['learning_rate']['value'])
        self.assertEqual(self.predictor.lmbda, hp_defaults['lmbda']['value'])
        self.assertEqual(self.predictor.loss_fn, hp_defaults['loss_fn']['value'])
        self.assertEqual(self.predictor.nn_shape['linear_stack'], self.mock_shape['linear_stack'])
    
    def test_dict_param_set_with_all_hps_configures_all_correctly(self):
        self.predictor.modify_hyper_parameter_values(param_set=self.hp_dict)
        
        self.assertEqual(self.predictor.mini_batch_size, self.hp_set.get('mini_batch_size').value)
        self.assertEqual(self.predictor.learning_rate, self.hp_set.get('learning_rate').value)
        self.assertEqual(self.predictor.lmbda, self.hp_set.get('lmbda').value)
        self.assertEqual(self.predictor.loss_fn, self.hp_set.get('loss_fn').value)
        self.assertEqual(self.predictor.nn_shape['linear_stack'], self.hp_set.get('linear_stack').value)

    def test_dict_param_set_with_some_hps_configures_all_correctly(self):
        partial_hp_dict = self.hp_dict
        del partial_hp_dict['lmbda']
        self.predictor.modify_hyper_parameter_values(param_set=partial_hp_dict)
        
        self.assertEqual(self.predictor.mini_batch_size, self.hp_set.get('mini_batch_size').value)
        self.assertEqual(self.predictor.learning_rate, self.hp_set.get('learning_rate').value)
        self.assertEqual(self.predictor.lmbda, hp_defaults['lmbda']['value'])
        self.assertEqual(self.predictor.loss_fn, self.hp_set.get('loss_fn').value)
        self.assertEqual(self.predictor.nn_shape['linear_stack'], self.hp_set.get('linear_stack').value)

    def test_no_param_set_input_configures_all_correctly(self):
        self.predictor.modify_hyper_parameter_values(param_set=None)
        
        self.assertEqual(self.predictor.mini_batch_size, hp_defaults['mini_batch_size']['value'])
        self.assertEqual(self.predictor.learning_rate, hp_defaults['learning_rate']['value'])
        self.assertEqual(self.predictor.lmbda, hp_defaults['lmbda']['value'])
        self.assertEqual(self.predictor.loss_fn, hp_defaults['loss_fn']['value'])
        self.assertEqual(self.predictor.nn_shape['linear_stack'], self.mock_shape['linear_stack'])
    
    def test_invalid_param_set_input_raises_error(self):
        with self.assertRaises(AttributeError):
            self.predictor.modify_hyper_parameter_values(param_set=[1,2])
        
    def tearDown(self):
        pass

class TestConfigureDataloader_NeuralNetPredictor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        self.mock_shape = {
            'players_input': 3,
            'teams_input': 2,
            'opps_input': 6,
            'stats_input': 8,
            'embedding_player': 2,
            'embedding_team': 4,
            'embedding_opp': 4,
            'linear_stack': 30,
            'stats_output': 3,
        }

        # Hyper-parameters
        self.mini_batch_size = 100
        self.hps = {'mini_batch_size': self.mini_batch_size}

        # Custom stats list (only using a subset of all statistics)
        self.scoring_weights = {'Pass Yds'   : 0.04,
                                'Rush Yds'   : 0.1,
                                'Rec Yds'    : 0.1
        }
        # Custom dataset
        self.dataset = StatsDataset(name='dataset',
                                    id_df=mock_data_predictors.id_df,
                                    pbp_df=mock_data_predictors.pbp_df_neural_net,
                                    boxscore_df=mock_data_predictors.bs_df)
        # Custom dataset 2
        pbp_df_modified = mock_data_predictors.pbp_df_neural_net.copy()
        pbp_df_modified['Rush Yds'] *= 2
        self.dataset2 = StatsDataset(name='dataset',
                                     id_df=mock_data_predictors.id_df,
                                     pbp_df=pbp_df_modified,
                                     boxscore_df=mock_data_predictors.bs_df)

        # Neural Net Predictor
        self.predictor = NeuralNetPredictor(name='test',
                                            nn_shape=self.mock_shape)
        self.predictor.modify_hyper_parameter_values(self.hps)

    def test_dataloaders_configured_from_datasets_correctly(self):
        training_data = self.dataset
        eval_data = self.dataset2
        torch.manual_seed(0) # Setting random seed to get the same data shuffle
        train_dataloader = self.predictor.configure_dataloader(training_data, mini_batch=True, shuffle=True)
        eval_dataloader = self.predictor.configure_dataloader(eval_data, mini_batch=False, shuffle=False)
        torch.manual_seed(0) # Setting random seed to get the same data shuffle
        expected_train_dataloader = DataLoader(training_data, batch_size=self.mini_batch_size, shuffle=True)
        expected_eval_dataloader = DataLoader(eval_data, batch_size=int(eval_data.x_data.shape[0]), shuffle=False)

        for (result, expected) in zip([train_dataloader, eval_dataloader], [expected_train_dataloader, expected_eval_dataloader]):
            self.assertTrue(result.dataset.equals(expected.dataset))
            self.assertEqual(result.batch_sampler.batch_size, expected.batch_sampler.batch_size)    
    
    def test_invalid_datasets_raises_error(self):
        with self.assertRaises(AttributeError):
            self.predictor.configure_dataloader([1,2,3])

    def tearDown(self):
        pass

class TestConfigureModelAndOptimizer_NeuralNetPredictor(unittest.TestCase):
    def setUp(self):

        # Hyper-parameters
        self.learning_rate = 1e4
        self.lmbda = 2
        self.linear_stack = 45
        self.hps = {'learning_rate': self.learning_rate,
                    'lmbda': self.lmbda,
                    'linear_stack': self.linear_stack}
        
        self.mock_shape = {
            'players_input': 3,
            'teams_input': 2,
            'opps_input': 6,
            'stats_input': 8,
            'embedding_player': 2,
            'embedding_team': 4,
            'embedding_opp': 4,
            'linear_stack': 30,
            'stats_output': 3,
        }
        self.mock_shape_modified = self.mock_shape.copy()
        self.mock_shape_modified['linear_stack'] = self.linear_stack

        # Custom stats list (only using a subset of all statistics)
        self.scoring_weights = {'Pass Yds'   : 0.04,
                                'Rush Yds'   : 0.1,
                                'Rec Yds'    : 0.1
        }

        # Neural Net Predictor
        self.predictor = NeuralNetPredictor(name='test',
                                            nn_shape=self.mock_shape)
    
    def test_updated_hp_values_result_in_correct_model_and_optimizer(self):
        self.predictor.modify_hyper_parameter_values(self.hps)
        self.predictor.configure_model_and_optimizer()
        expected_model = NeuralNetwork(shape=self.mock_shape_modified)
        expected_optimizer = torch.optim.SGD(self.predictor.model.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.lmbda)

        self.assertTrue(compare_net_sizes(self.predictor.model, expected_model))
        self.assertEqual(self.predictor.optimizer.state_dict(), expected_optimizer.state_dict())

    def test_wrong_hp_values_result_in_wrong_model_and_optimizer(self):
        self.predictor.configure_model_and_optimizer()
        expected_model = NeuralNetwork(shape=self.mock_shape_modified)
        expected_optimizer = torch.optim.SGD(self.predictor.model.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.lmbda)

        self.assertFalse(compare_net_sizes(self.predictor.model, expected_model))
        self.assertNotEqual(self.predictor.optimizer.state_dict(), expected_optimizer.state_dict())
    

    def tearDown(self):
        pass

class TestTrainAndValidate_NeuralNetPredictor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        
        self.save_folder = 'data/test files/empty/'

        self.mock_shape = {
            'players_input': 3,
            'teams_input': 2,
            'opps_input': 6,
            'stats_input': 8,
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
        self.training_data = StatsDataset(name='dataset',
                                    id_df=mock_data_predictors.id_df,
                                    pbp_df=mock_data_predictors.pbp_df_neural_net,
                                    boxscore_df=mock_data_predictors.bs_df)
        # Custom dataset 2
        pbp_df_modified = mock_data_predictors.pbp_df_neural_net.copy()
        pbp_df_modified['Rush Yds'] *= 2
        self.validation_data = StatsDataset(name='dataset',
                                     id_df=mock_data_predictors.id_df,
                                     pbp_df=pbp_df_modified,
                                     boxscore_df=mock_data_predictors.bs_df)
        # Hyper-parameters
        self.loss_fn = nn.MSELoss()
        self.loss_fn_hp = HyperParameter('loss_fn', optimizable=False, 
                                                 value=self.loss_fn)
        self.hp_set = HyperParameterSet((self.loss_fn_hp,),optimize=False)
        self.hp_dict = {'loss_fn': self.loss_fn}

        # Neural Net Predictor
        self.predictor = NeuralNetPredictor(name='test',
                                            save_folder=self.save_folder,
                                            nn_shape=self.mock_shape,
                                            max_epochs=1,
                                            n_epochs_to_stop=1)

        self.train_dataloader = self.predictor.configure_dataloader(self.training_data, mini_batch=True, shuffle=True)
        self.eval_dataloader = self.predictor.configure_dataloader(self.validation_data, mini_batch=False, shuffle=False)

    def test_input_dataloaders_runs_no_errors(self):
        self.predictor.train_and_validate(train_dataloader=self.train_dataloader, validation_dataloader=self.eval_dataloader,
                                              scoring_weights=self.scoring_weights)

    def test_input_datasets_runs_no_errors(self):
        self.predictor.train_and_validate(training_data=self.training_data, validation_data=self.validation_data,
                                          scoring_weights=self.scoring_weights)

    def test_input_both_dataloaders_and_datasets_runs_no_errors(self):
        self.predictor.train_and_validate(train_dataloader=self.train_dataloader, validation_dataloader=self.eval_dataloader,
                                          training_data=self.training_data, validation_data=self.validation_data,
                                          scoring_weights=self.scoring_weights)

    def test_input_both_dataloaders_and_datasets_prioritizes_dataloaders(self):
        self.predictor.train_and_validate(train_dataloader=self.train_dataloader, validation_dataloader=self.eval_dataloader,
                                          training_data=[1,2,6], validation_data='blah',
                                          scoring_weights=self.scoring_weights)

    def test_input_no_dataloaders_or_datasets_raises_error(self):
        with self.assertRaises(TypeError):
            self.predictor.train_and_validate(scoring_weights=self.scoring_weights)

    def test_invalid_dataloaders_raises_error(self):
        with self.assertRaises(AttributeError):
            self.predictor.train_and_validate(train_dataloader=self.training_data,
                                              validation_dataloader=self.validation_data,
                                              scoring_weights=self.scoring_weights)

    def test_invalid_datasets_raises_error(self):
        with self.assertRaises(AttributeError):
            self.predictor.train_and_validate(training_data=self.train_dataloader,
                                              validation_data=self.eval_dataloader,
                                              scoring_weights=self.scoring_weights)

    def test_return_value_types(self):
        final_val_perf, val_perfs = self.predictor.train_and_validate(self.train_dataloader,
                                                      self.eval_dataloader,
                                                      scoring_weights=self.scoring_weights)
        self.assertTrue(isinstance(final_val_perf,float))
        self.assertTrue(isinstance(val_perfs,list))
    
    def test_training_changes_model_parameters(self):
        # Save model before training
        self.predictor.save()
        # Train model
        self.predictor.train_and_validate(self.train_dataloader,
                                          self.eval_dataloader,
                                          scoring_weights=self.scoring_weights)
        post_trained_model = self.predictor.model
        # Load model from before training
        pre_trained_model = NeuralNetPredictor(name='loaded_copy',load_folder=self.save_folder).model

        self.assertTrue(check_all_model_layers_changed(pre_trained_model, post_trained_model))
    
    def test_max_training_iterations(self):
        n_epochs = 2
        self.predictor = NeuralNetPredictor(name='test',
                                            save_folder=self.save_folder,
                                            nn_shape=self.mock_shape,
                                            max_epochs=n_epochs,
                                            n_epochs_to_stop=n_epochs*2)

        _, val_perfs = self.predictor.train_and_validate(self.train_dataloader,
                                                      self.eval_dataloader,
                                                      scoring_weights=self.scoring_weights)
        self.assertEqual(len(val_perfs),n_epochs)
    
    def test_hyper_parameter_set_configures_all_correctly(self):
        try:
            self.predictor.train_and_validate(train_dataloader=self.train_dataloader,
                                            validation_dataloader=self.eval_dataloader,
                                            param_set=self.hp_set,
                                            scoring_weights=self.scoring_weights)
        except Exception as e:
            self.fail(f'Setting hyperparameters for train_and_validate raised {type(e)} unexpectedly.')
    
    def test_empty_hyper_parameter_set_configures_all_correctly(self):
        empty_hp_set = HyperParameterSet([],optimize=False)
        try:
            self.predictor.train_and_validate(train_dataloader=self.train_dataloader,
                                            validation_dataloader=self.eval_dataloader,
                                            param_set=empty_hp_set,
                                            scoring_weights=self.scoring_weights)
        except Exception as e:
            self.fail(f'Setting hyperparameters for train_and_validate raised {type(e)} unexpectedly.')
    
    def test_dict_param_set_configures_all_correctly(self):
        try:
            self.predictor.train_and_validate(train_dataloader=self.train_dataloader,
                                            validation_dataloader=self.eval_dataloader,
                                            param_set=self.hp_dict,
                                            scoring_weights=self.scoring_weights)
        except Exception as e:
            self.fail(f'Setting hyperparameters for train_and_validate raised {type(e)} unexpectedly.')

    def test_no_param_set_input_configures_all_correctly(self):
        try:
            self.predictor.train_and_validate(train_dataloader=self.train_dataloader,
                                            validation_dataloader=self.eval_dataloader,
                                            param_set=None,
                                            scoring_weights=self.scoring_weights)
        except Exception as e:
            self.fail(f'Setting hyperparameters for train_and_validate raised {type(e)} unexpectedly.')

    def tearDown(self):
        # Delete any files created in save folder
        for filename in os.listdir(self.save_folder):
            os.remove(self.save_folder+filename)
