import unittest
import numpy as np
# Class under test
from tuners.grid_search_tuner import GridSearchTuner
# Modules needed for test setup
from neural_net.hyper_parameter_set import HyperParameterSet
from neural_net.hyper_parameter import HyperParameter
import logging
import logging.config
from config.log_config import LOGGING_CONFIG

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')


# This is dumb, but need some dummy global variables for a test case
global_a = 0
global_b = 0

class TestConstructor_GridSearchTuner(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.hp1 = HyperParameter(name='hp1', value=4, optimizable=False)
        self.hp2 = HyperParameter(name='hp2', value=-1, optimizable=True, val_range=[-2.5,-0.5],val_scale='linear')
        self.hp3 = HyperParameter(name='hp3', value=100, optimizable=True, val_range=[1,10000],val_scale='log')
        self.hp_set = HyperParameterSet(hp_set=(self.hp1,self.hp2,self.hp3))

        self.expected_gridpoints = {'hp1':[4],
                                    'hp2':[-2.5, -2, -1.5, -1, -0.5],
                                    'hp3':[1, 10, 100, 1000, 10000]}
        self.expected_values = {'hp1':[4]*25,
                                'hp2':[-2.5, -2, -1.5, -1, -0.5,
                                       -2.5, -2, -1.5, -1, -0.5,
                                       -2.5, -2, -1.5, -1, -0.5,
                                       -2.5, -2, -1.5, -1, -0.5,
                                       -2.5, -2, -1.5, -1, -0.5],
                                'hp3':[1,1,1,1,1,
                                       10,10,10,10,10,
                                       100,100,100,100,100,
                                       1000,1000,1000,1000,1000,
                                       10000,10000,10000,10000,10000]}

        self.save_file = 'tests/_test_files/empty/hyper_tuner.csv'

        self.settings = {
            'optimize_hypers': True,
            'hyper_tuner_layers': 2,
            'hyper_tuner_steps_per_dim': 2,
            'plot_tuning_results': True,
        }

    def test_basic_attributes_no_optional_inputs(self):
        tuner = GridSearchTuner(self.hp_set, save_file=self.save_file)

        self.assertTrue(tuner.param_set.equals(self.hp_set))
        self.assertEqual(tuner.save_file, self.save_file)
        self.assertEqual(tuner.optimize_hypers, False)
        self.assertEqual(tuner.plot_tuning_results, False)
        self.assertEqual(tuner.hyper_tuner_layers, 1)
        self.assertEqual(tuner.hyper_tuner_steps_per_dim, 3)
        self.assertEqual(tuner.perf_list, [])
        self.assertEqual(tuner.hyper_tuning_table, [])

    def test_basic_attributes_all_optional_inputs(self):
        tuner = GridSearchTuner(self.hp_set, save_file=self.save_file, **self.settings)

        self.assertFalse(tuner.param_set.equals(self.hp_set)) # Not equal because values in param_set are modified by the tuner during init
        self.assertEqual(tuner.save_file, self.save_file)
        self.assertEqual(tuner.optimize_hypers, self.settings['optimize_hypers'])
        self.assertEqual(tuner.plot_tuning_results, self.settings['plot_tuning_results'])
        self.assertEqual(tuner.hyper_tuner_layers, self.settings['hyper_tuner_layers'])
        self.assertEqual(tuner.hyper_tuner_steps_per_dim, self.settings['hyper_tuner_steps_per_dim'])
        self.assertEqual(tuner.perf_list, [])
        self.assertEqual(tuner.hyper_tuning_table, [])

    def test_basic_attributes_some_optional_inputs(self):
        del self.settings['hyper_tuner_layers']
        tuner = GridSearchTuner(self.hp_set, save_file=self.save_file, **self.settings)

        self.assertFalse(tuner.param_set.equals(self.hp_set)) # Not equal because values in param_set are modified by the tuner during init
        self.assertEqual(tuner.save_file, self.save_file)
        self.assertEqual(tuner.optimize_hypers, self.settings['optimize_hypers'])
        self.assertEqual(tuner.plot_tuning_results, self.settings['plot_tuning_results'])
        self.assertEqual(tuner.hyper_tuner_layers, 1)
        self.assertEqual(tuner.hyper_tuner_steps_per_dim, self.settings['hyper_tuner_steps_per_dim'])
        self.assertEqual(tuner.perf_list, [])
        self.assertEqual(tuner.hyper_tuning_table, [])

    def test_missing_required_inputs_raises_error(self):
        with self.assertRaises(TypeError):
            GridSearchTuner(**self.settings)

    def test_gridpoints_and_value_combos_optimize_hypers_true(self):
        settings = {
            'optimize_hypers': True,
            'hyper_tuner_steps_per_dim': 5,
        }
        tuner = GridSearchTuner(self.hp_set, save_file=self.save_file, **settings)

        num_opt_hps = sum([1 if hp.optimizable else 0 for hp in self.hp_set.hyper_parameters])
        expected_combinations = settings['hyper_tuner_steps_per_dim']**(num_opt_hps)

        self.assertEqual(tuner.n_value_combinations, expected_combinations)
        for hp, hp_name in zip(tuner.param_set.hyper_parameters, self.expected_gridpoints):
            self.assertEqual(hp.gridpoints, self.expected_gridpoints[hp_name])
            self.assertEqual(hp.values, self.expected_values[hp_name])

    def test_gridpoints_and_value_combos_optimize_hypers_false(self):
        settings = {
            'optimize_hypers': False,
            'hyper_tuner_steps_per_dim': 5,
        }
        tuner = GridSearchTuner(self.hp_set, save_file=self.save_file, **settings)

        expected_combinations = 1

        self.assertEqual(tuner.n_value_combinations, expected_combinations)
        for hp in tuner.param_set.hyper_parameters:
            self.assertEqual(hp.values, [hp.value])
            with self.assertRaises(AttributeError):
                hp.gridpoints

    def test_gridpoints_and_value_combos_selection_hyperparameter(self):
        hp3 = HyperParameter(name='hp3', optimizable=True, value='a', val_range=['a','b','c'],val_scale='selection')
        hp_set = HyperParameterSet(hp_set=(self.hp1, self.hp2, hp3))
        settings = {
            'optimize_hypers': True,
            'hyper_tuner_steps_per_dim': 5,
        }
        tuner = GridSearchTuner(hp_set, save_file=self.save_file, **settings)

        expected_combinations = 15
        expected_gridpoints = {'hp1':[4],
                               'hp2':[-2.5, -2, -1.5, -1, -0.5],
                               'hp3':['a','b','c']}
        expected_values = {'hp1':[4]*expected_combinations,
                           'hp2':[-2.5, -2, -1.5, -1, -0.5,
                                  -2.5, -2, -1.5, -1, -0.5,
                                  -2.5, -2, -1.5, -1, -0.5,],
                           'hp3':['a','a','a','a','a',
                                  'b','b','b','b','b',
                                  'c','c','c','c','c']}        

        self.assertEqual(tuner.n_value_combinations, expected_combinations)
        for hp, hp_name in zip(tuner.param_set.hyper_parameters, expected_gridpoints):
            self.assertEqual(hp.gridpoints, expected_gridpoints[hp_name])
            self.assertEqual(hp.values, expected_values[hp_name])

    def test_gridpoints_and_value_combos_hyperparameter_with_val_scale_none(self):
        hp3 = HyperParameter(name='hp3', optimizable=True, value=5)
        hp_set = HyperParameterSet(hp_set=(self.hp1, self.hp2, hp3))
        settings = {
            'optimize_hypers': True,
            'hyper_tuner_steps_per_dim': 5,
        }
        tuner = GridSearchTuner(hp_set, save_file=self.save_file, **settings)

        expected_combinations = 5
        expected_gridpoints = {'hp1':[4],
                               'hp2':[-2.5, -2, -1.5, -1, -0.5],
                               'hp3':[5]}
        expected_values = {'hp1':[4]*expected_combinations,
                           'hp2':[-2.5, -2, -1.5, -1, -0.5],
                           'hp3':[5,5,5,5,5]}

        self.assertEqual(tuner.n_value_combinations, expected_combinations)
        for hp, hp_name in zip(tuner.param_set.hyper_parameters, expected_gridpoints):
            self.assertEqual(hp.gridpoints, expected_gridpoints[hp_name])
            self.assertEqual(hp.values, expected_values[hp_name])

    def test_gridpoints_and_value_combos_hyperparameter_with_invalid_val_scale_defaults_to_selection(self):
        hp3 = HyperParameter(name='hp3', optimizable=True, value=5, val_range=[1,10], val_scale=['xyz'])
        hp_set = HyperParameterSet(hp_set=(self.hp1, self.hp2, hp3))
        settings = {
            'optimize_hypers': True,
            'hyper_tuner_steps_per_dim': 5,
        }
        tuner = GridSearchTuner(hp_set, save_file=self.save_file, **settings)

        expected_combinations = 10
        expected_gridpoints = {'hp1':[4],
                               'hp2':[-2.5, -2, -1.5, -1, -0.5],
                               'hp3':[1,10]}
        expected_values = {'hp1':[4]*expected_combinations,
                           'hp2':[-2.5, -2, -1.5, -1, -0.5,
                                  -2.5, -2, -1.5, -1, -0.5],
                           'hp3':[1,1,1,1,1,
                                  10,10,10,10,10]}

        self.assertEqual(tuner.n_value_combinations, expected_combinations)
        for hp, hp_name in zip(tuner.param_set.hyper_parameters, expected_gridpoints):
            self.assertEqual(hp.gridpoints, expected_gridpoints[hp_name])
            self.assertEqual(hp.values, expected_values[hp_name])

    def test_gridpoints_and_value_combos_hyperparameter_with_invalid_val_range(self):
        hp3 = HyperParameter(name='hp3', optimizable=True, value=5, val_range=5, val_scale='linear')
        hp_set = HyperParameterSet(hp_set=(self.hp1, self.hp2, hp3))
        settings = {
            'optimize_hypers': True,
            'hyper_tuner_steps_per_dim': 5,
        }
        tuner = GridSearchTuner(hp_set, save_file=self.save_file, **settings)

        expected_combinations = 5
        expected_gridpoints = {'hp1':[4],
                               'hp2':[-2.5, -2, -1.5, -1, -0.5],
                               'hp3':[5]}
        expected_values = {'hp1':[4]*expected_combinations,
                           'hp2':[-2.5, -2, -1.5, -1, -0.5],
                           'hp3':[5,5,5,5,5]}

        self.assertEqual(tuner.n_value_combinations, expected_combinations)
        for hp, hp_name in zip(tuner.param_set.hyper_parameters, expected_gridpoints):
            self.assertEqual(hp.gridpoints, expected_gridpoints[hp_name])
            self.assertEqual(hp.values, expected_values[hp_name])

    # Tear Down
    def tearDown(self):
        pass

class TestTuneHyperParameters_GridSearchTuner(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.hp1 = HyperParameter(name='hp1', value=4, optimizable=False)
        self.hp2 = HyperParameter(name='hp2', value=-1, optimizable=True, val_range=[-2.5,-0.5],val_scale='linear')
        self.hp3 = HyperParameter(name='hp3', value=100, optimizable=True, val_range=[1,10000],val_scale='log')
        self.hp_set = HyperParameterSet(hp_set=(self.hp1,self.hp2,self.hp3))

        self.save_file = 'tests/_test_files/empty/tuner_test.csv'

        self.settings = {
            'optimize_hypers': True,
            'hyper_tuner_layers': 2,
            'hyper_tuner_steps_per_dim': 3,
            'plot_tuning_results': False,
        }

        self.tuner = GridSearchTuner(self.hp_set, save_file=self.save_file, **self.settings)

        self.eval_function = lambda param_set: param_set.get('hp1').value + 5*param_set.get('hp2').value + np.sqrt(param_set.get('hp3').value)
        self.max_val = 101.5
        self.min_val = -7.5

    def test_no_optional_inputs_returns_correct_value(self):
        min_val = self.tuner.tune_hyper_parameters(eval_function=self.eval_function)

        self.assertEqual(min_val, self.min_val)

    def test_finds_correct_value_to_zoom_in_on(self):
        eval_func = lambda param_set: (param_set.get('hp2').value - (-0.5))**2 + (param_set.get('hp3').value - np.sqrt(10))**2
        min_val = self.tuner.tune_hyper_parameters(eval_function=eval_func)

        expected_min_val = 0 # Will only be found by the computer if it refines grid correctly.

        self.assertEqual(min_val, expected_min_val)


    def test_perf_list_and_table_are_correct_length_for_one_tune_layer(self):
        self.tuner.hyper_tuner_layers = 1
        self.tuner.tune_hyper_parameters(eval_function=self.eval_function, save_function=None, reset_function=None,
                                         eval_kwargs=None, save_kwargs=None, reset_kwargs=None, maximize=True)

        perf_list_length = len(self.tuner.perf_list)
        table_rows = len(self.tuner.hyper_tuning_table)
        table_columns = len(self.tuner.hyper_tuning_table[0])
        expected_length = self.settings['hyper_tuner_steps_per_dim']**2 * self.tuner.hyper_tuner_layers
        expected_rows = expected_length * self.tuner.hyper_tuner_layers
        expected_columns = len(self.hp_set.hyper_parameters)+2

        self.assertEqual(perf_list_length, expected_length)
        self.assertEqual(table_rows, expected_rows)
        self.assertEqual(table_columns, expected_columns)

    def test_perf_list_and_table_are_correct_length_for_two_tune_layers(self):
        self.tuner.tune_hyper_parameters(eval_function=self.eval_function, save_function=None, reset_function=None,
                                         eval_kwargs=None, save_kwargs=None, reset_kwargs=None, maximize=True)

        perf_list_length = len(self.tuner.perf_list)
        table_rows = len(self.tuner.hyper_tuning_table)
        table_columns = len(self.tuner.hyper_tuning_table[0])
        expected_length = self.settings['hyper_tuner_steps_per_dim']**2
        expected_rows = expected_length * self.tuner.hyper_tuner_layers
        expected_columns = len(self.hp_set.hyper_parameters)+2

        self.assertEqual(perf_list_length, expected_length)
        self.assertEqual(table_rows, expected_rows)
        self.assertEqual(table_columns, expected_columns)

    def test_minimize_returns_correct_value(self):
        min_val = self.tuner.tune_hyper_parameters(eval_function=self.eval_function, maximize=False)

        self.assertEqual(min_val, self.min_val)

    def test_maximize_returns_correct_value(self):
        max_val = self.tuner.tune_hyper_parameters(eval_function=self.eval_function, maximize=True)

        self.assertEqual(max_val, self.max_val)

    def test_save_function_and_reset_function_handles_behave_correctly(self):

        def test_save():
            global global_a
            global_a += 1 # Counts number of times save function is called

        def test_reset():
            global global_b
            global_b += 1 # Counts number of times reset function is called

        self.tuner.tune_hyper_parameters(eval_function=self.eval_function,
                                         save_function=test_save,
                                         reset_function=test_reset)

        self.assertEqual(global_a, 2) # For given eval function, save should be called twice
        self.assertEqual(global_b, 1)

    def test_all_function_kwargs_passed_correctly(self):
        def test_eval(param_set, a=0):
            normal_func = param_set.get('hp1').value + 5*param_set.get('hp2').value + np.sqrt(param_set.get('hp3').value)
            return normal_func*a

        def test_save(b):
            global global_a
            global_a = b

        def test_reset(c):
            global global_b
            global_b = c

        a_val = 5
        b_val = 7
        c_val = -6

        min_val = self.tuner.tune_hyper_parameters(eval_function=test_eval, save_function=test_save, reset_function=test_reset,
                                         eval_kwargs={'a':a_val}, save_kwargs={'b':b_val}, reset_kwargs={'c':c_val})

        self.assertEqual(min_val, self.min_val*a_val)
        self.assertEqual(global_a, b_val)
        self.assertEqual(global_b, c_val)

    def test_improper_function_kwargs_raise_error(self):
        def test_eval(param_set, a=0):
            normal_func = param_set.get('hp1').value + 5*param_set.get('hp2').value + np.sqrt(param_set.get('hp3').value)
            return normal_func*a

        a_val = 5
        b_val = 7
    
        with self.assertRaises(TypeError):
            self.tuner.tune_hyper_parameters(eval_function=test_eval, 
                                             eval_kwargs={'a':a_val, 'b': b_val},)

    # Tear Down
    def tearDown(self):
        # Reset global variables
        global global_a
        global_a = 0
        global global_b
        global_b = 0

class TestRefineGrid_GridSearchTuner(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.hp1 = HyperParameter(name='hp1', value=4, optimizable=False)
        self.hp2 = HyperParameter(name='hp2', value=-1, optimizable=True, val_range=[-2.5,-0.5],val_scale='linear')
        self.hp3 = HyperParameter(name='hp3', value=100, optimizable=True, val_range=[1,10000],val_scale='log')
        self.hp_set = HyperParameterSet(hp_set=(self.hp1,self.hp2,self.hp3))

        self.save_file = 'tests/_test_files/empty/hyper_tuner.csv'

        self.settings = {
            'optimize_hypers': True,
            'hyper_tuner_layers': 2,
            'hyper_tuner_steps_per_dim': 3,
            'plot_tuning_results': False,
        }

        self.tuner = GridSearchTuner(self.hp_set, save_file=self.save_file, **self.settings)

        self.eval_function = lambda param_set: (param_set.get('hp2').value - (-0.5))**2 + (param_set.get('hp3').value - np.sqrt(10))**2
        self.max_val = 101.5
        self.min_val = -7.5

    def test_refines_to_correct_linear_and_log_gridpoints_and_values(self):
        optimal_ind = 2
        self.tuner.refine_grid(optimal_ind)

        expected_gridpoints = {'hp1':[4],
                               'hp2':[-1, -0.75, -0.5],
                               'hp3':[1, np.sqrt(10), 10]}
        expected_values = {'hp1':[4]*self.tuner.n_value_combinations,
                           'hp2':[-1, -0.75, -0.5,
                                  -1, -0.75, -0.5,
                                  -1, -0.75, -0.5],
                           'hp3':[1, 1, 1,
                                  np.sqrt(10), np.sqrt(10), np.sqrt(10),
                                  10, 10, 10]}

        for hp, hp_name in zip(self.tuner.param_set.hyper_parameters, expected_gridpoints):
            self.assertEqual(hp.gridpoints, expected_gridpoints[hp_name])
            self.assertEqual(hp.values, expected_values[hp_name])

    def test_refines_to_correct_selection_value(self):
        hp4 = HyperParameter(name='hp4', value='a', optimizable=True, val_range=['a','b'],val_scale='selection')
        hp_set = HyperParameterSet(hp_set=(self.hp1,self.hp2,self.hp3, hp4))
        tuner = GridSearchTuner(hp_set, save_file=self.save_file, **self.settings)

        optimal_ind = 2
        tuner.refine_grid(optimal_ind)

        expected_gridpoints = {'hp1':[4],
                               'hp2':[-1, -0.75, -0.5],
                               'hp3':[1, np.sqrt(10), 10],
                               'hp4':['a']}
        expected_values = {'hp1':[4]*9,
                           'hp2':[-1, -0.75, -0.5,
                                  -1, -0.75, -0.5,
                                  -1, -0.75, -0.5],
                           'hp3':[1, 1, 1,
                                  np.sqrt(10), np.sqrt(10), np.sqrt(10),
                                  10, 10, 10],
                           'hp4':['a']*9}

        for hp, hp_name in zip(tuner.param_set.hyper_parameters, expected_gridpoints):
            self.assertEqual(hp.gridpoints, expected_gridpoints[hp_name])
            self.assertEqual(hp.values, expected_values[hp_name])

    def test_refines_to_center_of_grid_correctly(self):
        optimal_ind = 4
        self.tuner.refine_grid(optimal_ind)

        expected_gridpoints = {'hp1':[4],
                               'hp2':[-2, -1.5, -1],
                               'hp3':[10, 100, 1000]}
        expected_values = {'hp1':[4]*self.tuner.n_value_combinations,
                           'hp2':[-2, -1.5, -1,
                                  -2, -1.5, -1,
                                  -2, -1.5, -1],
                           'hp3':[10, 10, 10,
                                  100, 100, 100,
                                  1000, 1000, 1000]}

        for hp, hp_name in zip(self.tuner.param_set.hyper_parameters, expected_gridpoints):
            self.assertEqual(hp.gridpoints, expected_gridpoints[hp_name])
            self.assertEqual(hp.values, expected_values[hp_name])

    # Tear Down
    def tearDown(self):
        pass
