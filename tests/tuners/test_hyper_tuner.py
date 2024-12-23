import unittest
import numpy as np
import pandas as pd
import pandas.testing as pdtest
import os
# Class under test
from tuners.hyper_tuner import HyperParamTuner
# Modules needed for test setup
from neural_net.hyper_parameter_set import HyperParameterSet
from neural_net.hyper_parameter import HyperParameter
from tuners.grid_search_tuner import GridSearchTuner
import logging
import logging.config
from config.log_config import LOGGING_CONFIG

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')

class TestConstructor_HyperParamTuner(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.hp1 = HyperParameter(name='hp1', value=4, optimizable=False)
        self.hp2 = HyperParameter(name='hp2', value=-1, optimizable=True, val_range=[-2,-0.5],val_scale='linear')
        self.hp_set = HyperParameterSet(hp_set=(self.hp1,self.hp2))

        self.save_file = 'data/test files/empty/hyper_tuner.csv'

    def test_basic_attributes_no_optional_inputs(self):
        tuner = HyperParamTuner(self.hp_set, save_file=self.save_file)

        self.assertTrue(tuner.param_set.equals(self.hp_set))
        self.assertEqual(tuner.save_file, self.save_file)
        self.assertEqual(tuner.optimize_hypers, False)
        self.assertEqual(tuner.plot_tuning_results, False)
        self.assertEqual(tuner.perf_list, [])
        self.assertEqual(tuner.hyper_tuning_table, [])

    def test_basic_attributes_all_optional_inputs(self):
        tuner = HyperParamTuner(self.hp_set, save_file=self.save_file, optimize_hypers=True, plot_tuning_results=True)

        self.assertTrue(tuner.param_set.equals(self.hp_set))
        self.assertEqual(tuner.save_file, self.save_file)
        self.assertEqual(tuner.optimize_hypers, True)
        self.assertEqual(tuner.plot_tuning_results, True)
        self.assertEqual(tuner.perf_list, [])
        self.assertEqual(tuner.hyper_tuning_table, [])

    def test_missing_inputs_raises_error(self):
        with self.assertRaises(TypeError):
            HyperParamTuner(save_file=self.save_file)

    # Tear Down
    def tearDown(self):
        pass

class TestRefineAreaOfInterest_HyperParamTuner(unittest.TestCase):
    def setUp(self):
        self.hp1 = HyperParameter(name='hp1', value=4, optimizable=False)
        self.hp2 = HyperParameter(name='hp2', value=-1, optimizable=True, val_range=[-2.5,-0.5],val_scale='linear')
        self.hp3 = HyperParameter(name='hp3', value=100, optimizable=True, val_range=[1,10000],val_scale='log')
        self.hp_set = HyperParameterSet(hp_set=(self.hp1,self.hp2,self.hp3))

        self.save_file = 'data/test files/empty/hyper_tuner.csv'

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

    def test_refines_to_correct_linear_and_log_val_ranges_for_grid_tuner(self):
        optimal_ind = 2
        self.tuner.refine_area_of_interest(optimal_ind)

        expected_val_range = {'hp1':[4],
                               'hp2':[-1, -0.5],
                               'hp3':[1, 10]}

        for hp, hp_name in zip(self.tuner.param_set.hyper_parameters, expected_val_range):
            self.assertEqual(hp.val_range, expected_val_range[hp_name])

    def test_refines_to_correct_selection_value(self):
        hp4 = HyperParameter(name='hp4', value='a', optimizable=True, val_range=['a','b'],val_scale='selection')
        hp_set = HyperParameterSet(hp_set=(self.hp1,self.hp2,self.hp3, hp4))
        tuner = GridSearchTuner(hp_set, save_file=self.save_file, **self.settings)

        optimal_ind = 2
        tuner.refine_area_of_interest(optimal_ind)

        expected_val_ranges = {'hp1':[4],
                               'hp2':[-1, -0.5],
                               'hp3':[1, 10],
                               'hp4':['a']}

        for hp, hp_name in zip(tuner.param_set.hyper_parameters, expected_val_ranges):
            self.assertEqual(hp.val_range, expected_val_ranges[hp_name])

    def tearDown(self):
        pass

class TestSaveHPTuningResults_HyperParamTuner(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.save_folder = 'data/test files/empty/'
        self.save_file = self.save_folder + 'tuner_test.csv'
        self.df_columns = ['hp1','hp2','Model Performance']

        self.values_hp1 = [4,5,6,7]
        self.values_hp2 = [-3,-2,-1,-0.5]

        self.hp1 = HyperParameter(name='hp1', value=4, optimizable=False)
        self.hp2 = HyperParameter(name='hp2', value=-1, optimizable=True, val_range=[-2,-0.5],val_scale='linear')
        self.hp1.values = self.values_hp1
        self.hp2.values = self.values_hp2

        self.hp_set = HyperParameterSet(hp_set=(self.hp1,self.hp2))
        self.tuner = HyperParamTuner(self.hp_set, save_file=self.save_file)
        
        self.tuner.perf_list = [0,1,2,3]

    def test_save_without_perf_data_returns_empty_dataframe(self):
        self.tuner.perf_list = []
        df = self.tuner._save_hp_tuning_results()
        empty_df = pd.DataFrame(columns=self.df_columns)

        pdtest.assert_frame_equal(df,empty_df)

    def test_save_with_perf_data_returns_populated_dataframe(self):
        df = self.tuner._save_hp_tuning_results()
        
        expected_df = pd.DataFrame(data=[self.hp1.values,self.hp2.values,self.tuner.perf_list],
                                   index=self.df_columns).transpose()

        pdtest.assert_frame_equal(df,expected_df, check_dtype=False)

    def test_save_additional_columns_with_list_of_data(self):
        addl_data = [10, 20, 30, 40]
        self.df_columns.append('foo')
        df = self.tuner._save_hp_tuning_results(addl_columns={'foo': addl_data})
        
        expected_df = pd.DataFrame(data=[self.hp1.values,self.hp2.values,self.tuner.perf_list, addl_data],
                                   index=self.df_columns).transpose()

        pdtest.assert_frame_equal(df,expected_df, check_dtype=False)

    def test_save_additional_columns_with_single_number_works(self):
        addl_data = 10
        self.df_columns.append('foo')
        df = self.tuner._save_hp_tuning_results(addl_columns={'foo': addl_data})
        
        addl_data_list = [addl_data]*4
        expected_df = pd.DataFrame(data=[self.hp1.values,self.hp2.values,self.tuner.perf_list, addl_data_list],
                                   index=self.df_columns).transpose()

        pdtest.assert_frame_equal(df,expected_df, check_dtype=False)

    def test_save_additional_columns_with_list_of_wrong_length_raises_error(self):
        addl_data = [10,20,30]

        with self.assertRaises(ValueError):
            self.tuner._save_hp_tuning_results(addl_columns={'foo': addl_data})

    def test_save_to_file_works_as_expected(self):
        self.tuner._save_hp_tuning_results(filename=self.save_file)
        
        df = pd.read_csv(self.save_file, index_col=0)
        
        expected_df = pd.DataFrame(data=[self.hp1.values,self.hp2.values,self.tuner.perf_list],
                                   index=self.df_columns).transpose()

        pdtest.assert_frame_equal(df,expected_df, check_dtype=False)

    def test_save_with_prior_data_appends_new_data(self):
        self.tuner._save_hp_tuning_results(filename=self.save_file)
        self.tuner._save_hp_tuning_results(filename=self.save_file)
        
        df = pd.read_csv(self.save_file, index_col=0)
        
        one_df = pd.DataFrame(data=[self.hp1.values,self.hp2.values,self.tuner.perf_list],
                                   index=self.df_columns).transpose()
        expected_df = pd.concat((one_df,one_df)).reset_index(drop=True)

        pdtest.assert_frame_equal(df,expected_df, check_dtype=False)
    
    # Tear Down
    def tearDown(self):
        # Delete any files created in save folder
        for filename in os.listdir(self.save_folder):
            os.remove(self.save_folder+filename)

