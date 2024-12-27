import unittest
import numpy as np
# Class under test
from tuners.recursive_random_search_tuner import RecursiveRandomSearchTuner
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

def calc_r_percentile(n,p):
    r = 1 - (np.e ** (np.log(1 - p) / n))
    return r
def calc_n_samples(r,p):
    n = int(np.log(1 - p) / np.log(1 - r))+1
    return n

class TestConstructor_RecRandomSearchTuner(unittest.TestCase):
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

        self.save_file = 'data/test files/empty/hyper_tuner.csv'

        self.base_settings = {
            'optimize_hypers': True,
            'plot_tuning_results': True,
        }
        
        self.default_settings = {
            'p_conf': 0.99,
            'q_conf': 0.99,
            'v_expect_imp': None,
            'c_shrink_ratio': 0.5,
            's_shrink_thresh': 0.001,
            'n_explore_samples': 50,
            'l_exploit_samples': 20
        }
        self.def_r_percentile = calc_r_percentile(self.default_settings['n_explore_samples'],
                                                  self.default_settings['p_conf'])

        self.custom_settings = {
            'p_conf': 0.8,
            'q_conf': 0.6,
            'v_expect_imp': None,
            'c_shrink_ratio': 0.2,
            's_shrink_thresh': 0.01,
            'n_explore_samples': 320,
            'l_exploit_samples': 39,
            'max_samples': 100
        }
        self.custom_r_percentile = calc_r_percentile(self.custom_settings['n_explore_samples'],
                                                  self.custom_settings['p_conf'])
        self.custom_n_value_combinations = min(
            self.custom_settings['n_explore_samples'],
            self.custom_settings['max_samples']
        )

    def test_basic_attributes_not_optimizing_no_optional_inputs(self):
        tuner = RecursiveRandomSearchTuner(self.hp_set,
                                           save_file=self.save_file)

        self.assertTrue(tuner.param_set.equals(self.hp_set))
        self.assertEqual(tuner.save_file, self.save_file)
        self.assertEqual(tuner.optimize_hypers, False)
        self.assertEqual(tuner.plot_tuning_results, False)
        self.assertEqual(tuner.n_value_combinations, 1)
        self.assertEqual(tuner.perf_list, [])
        self.assertEqual(tuner.hyper_tuning_table, [])

        self.assertEqual(tuner.l_exploit_samples, self.default_settings['l_exploit_samples'])
        self.assertEqual(tuner.c_shrink_ratio, self.default_settings['c_shrink_ratio'])
        self.assertEqual(tuner.s_shrink_thresh, self.default_settings['s_shrink_thresh'])
        self.assertEqual(tuner.r_percentile, self.def_r_percentile)
        self.assertEqual(tuner.max_samples, 10*self.default_settings['n_explore_samples'])

    def test_basic_attributes_optimizing_no_optional_inputs(self):
        tuner = RecursiveRandomSearchTuner(self.hp_set,
                                           save_file=self.save_file,
                                           **self.base_settings)

        self.assertFalse(tuner.param_set.equals(self.hp_set)) # Not equal because values in param_set are modified by the tuner during init
        self.assertEqual(tuner.save_file, self.save_file)
        self.assertEqual(tuner.optimize_hypers, True)
        self.assertEqual(tuner.plot_tuning_results, True)
        self.assertEqual(tuner.perf_list, [])
        self.assertEqual(tuner.hyper_tuning_table, [])
        self.assertEqual(tuner.n_value_combinations, self.default_settings['n_explore_samples'])
        self.assertEqual(tuner.l_exploit_samples, self.default_settings['l_exploit_samples'])
        self.assertEqual(tuner.c_shrink_ratio, self.default_settings['c_shrink_ratio'])
        self.assertEqual(tuner.s_shrink_thresh, self.default_settings['s_shrink_thresh'])
        self.assertEqual(tuner.r_percentile, self.def_r_percentile)
        self.assertEqual(tuner.max_samples, 10*self.default_settings['n_explore_samples'])

    def test_basic_attributes_all_optional_inputs(self):
        tuner = RecursiveRandomSearchTuner(self.hp_set, save_file=self.save_file, 
                                           **self.base_settings, **self.custom_settings)

        self.assertFalse(tuner.param_set.equals(self.hp_set)) # Not equal because values in param_set are modified by the tuner during init
        self.assertEqual(tuner.save_file, self.save_file)
        self.assertEqual(tuner.optimize_hypers, self.base_settings['optimize_hypers'])
        self.assertEqual(tuner.plot_tuning_results, self.base_settings['plot_tuning_results'])
        self.assertEqual(tuner.perf_list, [])
        self.assertEqual(tuner.hyper_tuning_table, [])
        self.assertEqual(tuner.n_value_combinations, self.custom_n_value_combinations)
        self.assertEqual(tuner.l_exploit_samples, self.custom_settings['l_exploit_samples'])
        self.assertEqual(tuner.c_shrink_ratio, self.custom_settings['c_shrink_ratio'])
        self.assertEqual(tuner.s_shrink_thresh, self.custom_settings['s_shrink_thresh'])
        self.assertEqual(tuner.r_percentile, self.custom_r_percentile)
        self.assertEqual(tuner.max_samples, self.custom_settings['max_samples'])

    def test_basic_attributes_some_optional_inputs(self):
        del self.custom_settings['l_exploit_samples']
        del self.custom_settings['c_shrink_ratio']
        tuner = RecursiveRandomSearchTuner(self.hp_set, save_file=self.save_file, 
                                           **self.base_settings, **self.custom_settings)

        self.assertFalse(tuner.param_set.equals(self.hp_set)) # Not equal because values in param_set are modified by the tuner during init
        self.assertEqual(tuner.save_file, self.save_file)
        self.assertEqual(tuner.optimize_hypers, self.base_settings['optimize_hypers'])
        self.assertEqual(tuner.plot_tuning_results, self.base_settings['plot_tuning_results'])
        self.assertEqual(tuner.perf_list, [])
        self.assertEqual(tuner.hyper_tuning_table, [])
        self.assertEqual(tuner.n_value_combinations, self.custom_n_value_combinations)
        self.assertEqual(tuner.l_exploit_samples, self.default_settings['l_exploit_samples'])
        self.assertEqual(tuner.c_shrink_ratio, self.default_settings['c_shrink_ratio'])
        self.assertEqual(tuner.s_shrink_thresh, self.custom_settings['s_shrink_thresh'])
        self.assertEqual(tuner.r_percentile, self.custom_r_percentile)
        self.assertEqual(tuner.max_samples, self.custom_settings['max_samples'])

    def test_missing_required_inputs_raises_error(self):
        with self.assertRaises(TypeError):
            RecursiveRandomSearchTuner(**self.base_settings)

    def test_r_percentile_input_overrides_n_explore_samples(self):
        self.custom_settings['r_percentile'] = 0.2
        del self.custom_settings['max_samples']
        expected_n = calc_n_samples(self.custom_settings['r_percentile'],
                                    self.custom_settings['p_conf'])

        tuner = RecursiveRandomSearchTuner(self.hp_set, save_file=self.save_file, 
                                           **self.base_settings, **self.custom_settings)

        self.assertEqual(tuner.n_value_combinations, expected_n)
        self.assertEqual(tuner.r_percentile, self.custom_settings['r_percentile'])
        self.assertEqual(tuner.max_samples, expected_n*10)

    def test_v_expect_imp_input_overrides_l_exploit_samples(self):
        self.custom_settings['v_expect_imp'] = 0.4
        del self.custom_settings['max_samples']
        expected_l = calc_n_samples(self.custom_settings['v_expect_imp'],
                                    self.custom_settings['q_conf'])

        tuner = RecursiveRandomSearchTuner(self.hp_set, save_file=self.save_file, 
                                           **self.base_settings, **self.custom_settings)

        self.assertEqual(tuner.l_exploit_samples, expected_l)

    def test_hp_values_optimize_true(self):
        tuner = RecursiveRandomSearchTuner(self.hp_set, save_file=self.save_file, 
                                           **self.base_settings, **self.default_settings)

        expected_len = self.default_settings['n_explore_samples']

        for hp in tuner.param_set.hyper_parameters:
            self.assertEqual(len(hp.values),expected_len)
            self.assertTrue(all(np.array(hp.values) <= max(hp.val_range)))
            self.assertTrue(all(np.array(hp.values) >= min(hp.val_range)))

    def test_hp_values_optimize_false(self):
        tuner = RecursiveRandomSearchTuner(self.hp_set, save_file=self.save_file)

        for hp in tuner.param_set.hyper_parameters:
            self.assertEqual([hp.value],hp.values)

    # Tear Down
    def tearDown(self):
        pass

class TestTuneHyperParameters_RecRandomSearchTuner(unittest.TestCase):
    # Set Up
    def setUp(self):
        np.random.seed(0)
        
        self.hp1 = HyperParameter(name='hp1', value=4, optimizable=False)
        self.hp2 = HyperParameter(name='hp2', value=-1, optimizable=True, val_range=[-2.5,-0.5],val_scale='linear')
        self.hp3 = HyperParameter(name='hp3', value=100, optimizable=True, val_range=[1,10000],val_scale='log')
        self.hp_set = HyperParameterSet(hp_set=(self.hp1,self.hp2,self.hp3))

        self.save_file = 'data/test files/empty/tuner_test.csv'

        self.settings = {
            'optimize_hypers': True,
            'plot_tuning_results': False,
            'max_samples': 100
        }

        self.tuner = RecursiveRandomSearchTuner(self.hp_set, save_file=self.save_file, **self.settings)

        self.eval_function = lambda param_set: param_set.get('hp1').value + 5*param_set.get('hp2').value + np.sqrt(param_set.get('hp3').value)
        self.max_val = 92.3143932 # Max value found by this algo with the manually set random seed
        self.min_val = -6.113715642 # Min value found by this algo with the manually set random seed

    def test_no_optional_inputs_returns_correct_value(self):
        min_val = self.tuner.tune_hyper_parameters(eval_function=self.eval_function)

        self.assertAlmostEqual(min_val, self.min_val)

    def test_perf_list_and_table_are_correct_length_exploration_only(self):
        self.tuner.n_value_combinations -= 1
        self.tuner.tune_hyper_parameters(eval_function=self.eval_function, save_function=None, reset_function=None,
                                         eval_kwargs=None, save_kwargs=None, reset_kwargs=None, maximize=True)

        perf_list_length = len(self.tuner.perf_list)
        table_rows = len(self.tuner.hyper_tuning_table)
        table_columns = len(self.tuner.hyper_tuning_table[0])
        expected_length = self.tuner.n_value_combinations
        expected_columns = len(self.hp_set.hyper_parameters)+1

        self.assertEqual(perf_list_length, expected_length)
        self.assertEqual(table_rows, expected_length)
        self.assertEqual(table_columns, expected_columns)

    def test_perf_list_and_table_are_correct_length_with_exploitation(self):
        self.tuner.max_samples = self.tuner.n_value_combinations + 10
        self.tuner.tune_hyper_parameters(eval_function=self.eval_function, save_function=None, reset_function=None,
                                         eval_kwargs=None, save_kwargs=None, reset_kwargs=None, maximize=True)

        perf_list_length = len(self.tuner.perf_list)
        table_rows = len(self.tuner.hyper_tuning_table)
        table_columns = len(self.tuner.hyper_tuning_table[0])
        expected_length = self.tuner.max_samples
        expected_columns = len(self.hp_set.hyper_parameters)+1

        self.assertEqual(perf_list_length, expected_length)
        self.assertEqual(table_rows, expected_length)
        self.assertEqual(table_columns, expected_columns)

    def test_minimize_returns_correct_value(self):
        min_val = self.tuner.tune_hyper_parameters(eval_function=self.eval_function, maximize=False)

        self.assertAlmostEqual(min_val, self.min_val)

    def test_maximize_returns_correct_value(self):
        max_val = self.tuner.tune_hyper_parameters(eval_function=self.eval_function, maximize=True)

        self.assertAlmostEqual(max_val, self.max_val)

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

        self.assertEqual(global_a, 6) # For given eval function, save should be called 6 times
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

        self.assertAlmostEqual(min_val, self.min_val*a_val)
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
