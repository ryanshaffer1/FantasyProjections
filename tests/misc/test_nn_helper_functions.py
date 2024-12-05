import unittest
import pandas as pd
import pandas.testing as pdtest
import torch
from misc.nn_helper_functions import normalize_stat, unnormalize_stat, stats_to_fantasy_points
from config import stats_config

# Set up same logger as project code
import logging
import logging.config
from config.log_config import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')



class TestNormalizeStat_DefaultThresholds(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.thresholds = stats_config.default_norm_thresholds
        self.column_names = list(self.thresholds.keys())
        # Set various bounds on values based on threshold mins and maxes
        self.threshold_mins = [x[0] for x in self.thresholds.values()]
        self.threshold_maxes = [x[1] for x in self.thresholds.values()]
        self.threshold_midpoints = [0.5*(x[0]+x[1]) for x in self.thresholds.values()]

    # Tests of normalization calculations across input ranges
    def test_min_thresholds_normalize_to_zero(self):
        result = normalize_stat(pd.DataFrame(data=self.threshold_mins, index=self.column_names).T)
        all_zeros = pd.DataFrame(data=[float(0)]*len(self.column_names), index=self.column_names).T
        pdtest.assert_frame_equal(result, all_zeros, check_dtype=False)

    def test_below_min_thresholds_normalize_to_zero(self):
        result = normalize_stat((pd.DataFrame(data=self.threshold_mins, index=self.column_names).T) - 10)
        all_zeros = pd.DataFrame(data=[float(0)]*len(self.column_names), index=self.column_names).T
        pdtest.assert_frame_equal(result, all_zeros, check_dtype=False)

    def test_max_thresholds_normalize_to_one(self):
        result = normalize_stat(pd.DataFrame(data=self.threshold_maxes, index=self.column_names).T)
        all_ones = pd.DataFrame(data=[float(1)]*len(self.column_names), index=self.column_names).T
        pdtest.assert_frame_equal(result, all_ones, check_dtype=False)

    def test_above_min_thresholds_normalize_to_one(self):
        result = normalize_stat((pd.DataFrame(data=self.threshold_maxes, index=self.column_names).T) + 10)
        all_ones = pd.DataFrame(data=[float(1)]*len(self.column_names), index=self.column_names).T
        pdtest.assert_frame_equal(result, all_ones, check_dtype=False)

    def test_midway_values_normalize_to_one_half(self):
        result = normalize_stat(pd.DataFrame(data=self.threshold_midpoints, index=self.column_names).T)
        all_point_fives = pd.DataFrame(data=[0.5]*len(self.column_names), index=self.column_names).T
        pdtest.assert_frame_equal(result, all_point_fives, check_dtype=False)

    # Test input data types

    def test_list_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            normalize_stat(self.threshold_midpoints)
    
    def test_single_stat_series_input_passes(self):
        result = normalize_stat(pd.Series(data=[self.threshold_midpoints[0]]*5, name=self.column_names[0]))
        all_point_fives = pd.Series(data=[0.5]*5, name=self.column_names[0])
        pdtest.assert_series_equal(result, all_point_fives, check_dtype=False)

    # Tear Down
    def tearDown(self):
        pass

class TestNormalizeStat_CustomThresholds(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.thresholds = {'StatA':[-1, 1],
                           'StatB':[100,200],
                           'StatC':[-50,10000]}
        self.column_names = list(self.thresholds.keys())
        # Set various bounds on values based on threshold mins and maxes
        self.threshold_mins = [x[0] for x in self.thresholds.values()]
        self.threshold_maxes = [x[1] for x in self.thresholds.values()]
        self.threshold_midpoints = [0.5*(x[0]+x[1]) for x in self.thresholds.values()]

    # Tests of normalization calculations across input ranges
    def test_min_thresholds_normalize_to_zero(self):
        result = normalize_stat(pd.DataFrame(data=self.threshold_mins, index=self.column_names).T, 
                                thresholds=self.thresholds)
        all_zeros = pd.DataFrame(data=[float(0)]*len(self.column_names), index=self.column_names).T
        pdtest.assert_frame_equal(result, all_zeros, check_dtype=False)

    def test_below_min_thresholds_normalize_to_zero(self):
        result = normalize_stat((pd.DataFrame(data=self.threshold_mins, index=self.column_names).T) - 10,
                                thresholds=self.thresholds)
        all_zeros = pd.DataFrame(data=[float(0)]*len(self.column_names), index=self.column_names).T
        pdtest.assert_frame_equal(result, all_zeros, check_dtype=False)

    def test_max_thresholds_normalize_to_one(self):
        result = normalize_stat(pd.DataFrame(data=self.threshold_maxes, index=self.column_names).T, 
                                thresholds=self.thresholds)
        all_ones = pd.DataFrame(data=[float(1)]*len(self.column_names), index=self.column_names).T
        pdtest.assert_frame_equal(result, all_ones, check_dtype=False)

    def test_above_min_thresholds_normalize_to_one(self):
        result = normalize_stat((pd.DataFrame(data=self.threshold_maxes, index=self.column_names).T) + 10,
                                thresholds=self.thresholds)
        all_ones = pd.DataFrame(data=[float(1)]*len(self.column_names), index=self.column_names).T
        pdtest.assert_frame_equal(result, all_ones, check_dtype=False)

    def test_midway_values_normalize_to_one_half(self):
        result = normalize_stat(pd.DataFrame(data=self.threshold_midpoints, index=self.column_names).T, 
                                thresholds=self.thresholds)
        all_point_fives = pd.DataFrame(data=[0.5]*len(self.column_names), index=self.column_names).T
        pdtest.assert_frame_equal(result, all_point_fives, check_dtype=False)

    # Tear Down
    def tearDown(self):
        pass
    
class TestUnNormalizeStat_DefaultThresholds(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.thresholds = stats_config.default_norm_thresholds
        self.column_names = list(self.thresholds.keys())
        # Set various bounds on values based on threshold mins and maxes
        self.threshold_mins = [x[0] for x in self.thresholds.values()]
        self.threshold_maxes = [x[1] for x in self.thresholds.values()]
        self.threshold_midpoints = [0.5*(x[0]+x[1]) for x in self.thresholds.values()]

    # Tests of normalization calculations across input ranges
    # Note: No test of below min or above max thresholds, since un-normalization cannot produce values outside of thresholds

    def test_zero_unnormalizes_to_min_threshold(self):
        result = unnormalize_stat(pd.DataFrame(data=[0]*len(self.column_names), index=self.column_names).T)
        min_thresholds = pd.DataFrame(data=self.threshold_mins, index=self.column_names).T
        pdtest.assert_frame_equal(result, min_thresholds, check_dtype=False)

    def test_one_unnormalizes_to_max_threshold(self):
        result = unnormalize_stat(pd.DataFrame(data=[1]*len(self.column_names), index=self.column_names).T)
        threshold_maxes = pd.DataFrame(data=self.threshold_maxes, index=self.column_names).T
        pdtest.assert_frame_equal(result, threshold_maxes, check_dtype=False)

    def test_one_half_unnormalizes_to_threshold_midpoint(self):
        result = unnormalize_stat(pd.DataFrame(data=[0.5]*len(self.column_names), index=self.column_names).T)
        threshold_midpoints = pd.DataFrame(data=self.threshold_midpoints, index=self.column_names).T
        pdtest.assert_frame_equal(result, threshold_midpoints, check_dtype=False)

    # Test input data types

    def test_list_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            unnormalize_stat([0.5]*5)
    
    def test_single_stat_series_input_passes(self):
        result = unnormalize_stat(pd.Series(data=[0.5]*5, name=self.column_names[0]))
        threshold_midpoints = pd.Series(data=[self.threshold_midpoints[0]]*5, name=self.column_names[0])
        pdtest.assert_series_equal(result, threshold_midpoints, check_dtype=False)

    # Tear Down
    def tearDown(self):
        pass

class TestUnNormalizeStat_CustomThresholds(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.thresholds = {'StatA':[-1, 1],
                           'StatB':[100,200],
                           'StatC':[-50,10000]}
        self.column_names = list(self.thresholds.keys())
        # Set various bounds on values based on threshold mins and maxes
        self.threshold_mins = [x[0] for x in self.thresholds.values()]
        self.threshold_maxes = [x[1] for x in self.thresholds.values()]
        self.threshold_midpoints = [0.5*(x[0]+x[1]) for x in self.thresholds.values()]

    # Tests of normalization calculations across input ranges
    def test_zero_unnormalizes_to_min_threshold(self):
        result = unnormalize_stat(pd.DataFrame(data=[0]*len(self.column_names), index=self.column_names).T, 
                                thresholds=self.thresholds)
        min_thresholds = pd.DataFrame(data=self.threshold_mins, index=self.column_names).T
        pdtest.assert_frame_equal(result, min_thresholds, check_dtype=False)

    # No test of below min threshold, since un-normalization cannot produce values outside of thresholds

    def test_one_unnormalizes_to_max_threshold(self):
        result = unnormalize_stat(pd.DataFrame(data=[1]*len(self.column_names), index=self.column_names).T, 
                                thresholds=self.thresholds)
        threshold_maxes = pd.DataFrame(data=self.threshold_maxes, index=self.column_names).T
        pdtest.assert_frame_equal(result, threshold_maxes, check_dtype=False)

    # No test of above max threshold, since un-normalization cannot produce values outside of thresholds

    def test_one_half_unnormalizes_to_threshold_midpoint(self):
        result = unnormalize_stat(pd.DataFrame(data=[0.5]*len(self.column_names), index=self.column_names).T, 
                                thresholds=self.thresholds)
        threshold_midpoints = pd.DataFrame(data=self.threshold_midpoints, index=self.column_names).T
        pdtest.assert_frame_equal(result, threshold_midpoints, check_dtype=False)

    # Tear Down
    def tearDown(self):
        pass

class TestNormalize_to_UnNormalize(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.thresholds = {'StatA':[-1, 1],
                           'StatB':[100,200],
                           'StatC':[-50,10000]}
        self.column_names = list(self.thresholds.keys())
        # Set various bounds on values based on threshold mins and maxes
        self.threshold_midpoints = [0.5*(x[0]+x[1]) for x in self.thresholds.values()]

    # Tests of normalization calculations across input ranges
    def test_norm_to_unnorm_produces_original_numbers_within_threshold_range(self):
        original_df = pd.DataFrame(data=self.threshold_midpoints, index=self.column_names).T
        normalized_df = normalize_stat(original_df, thresholds=self.thresholds)
        unnormalized_df = unnormalize_stat(normalized_df, thresholds=self.thresholds)
        pdtest.assert_frame_equal(original_df, unnormalized_df, check_dtype=False)

    # Tear Down
    def tearDown(self):
        pass

class TestStatsToFantasyPoints_DefaultWeights(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.weights = stats_config.default_scoring_weights
        self.thresholds = stats_config.default_norm_thresholds
        self.stat_names = list(self.weights.keys())
        self.sum_of_weights = sum(self.weights.values())
        # Various inputs for test methods
        self.one_df = pd.DataFrame(data=[1]*len(self.stat_names), index=self.stat_names).T
        self.one_tensor = torch.tensor(self.one_df.values)
        self.one_df_normalized = normalize_stat(self.one_df, thresholds=self.thresholds)
        # Correct result for various inputs
        self.one_result = self.one_df.copy()
        self.one_result['Fantasy Points'] = self.sum_of_weights

    def test_zero_stats_equals_zero_points(self):
        # Custom setup
        zero_df = pd.DataFrame(data=[0]*len(self.stat_names), index=self.stat_names).T
        zero_result = zero_df.copy()
        zero_result['Fantasy Points'] = 0

        pdtest.assert_frame_equal(stats_to_fantasy_points(zero_df), 
                                  zero_result, check_dtype=False)

    def test_one_of_each_stat_equals_correct_points(self):
        pdtest.assert_frame_equal(stats_to_fantasy_points(self.one_df),
                                  self.one_result, check_dtype=False)

    def test_negative_one_of_each_stat_equals_correct_points(self):
        # Custom setup
        minus_one_df = self.one_df*(-1)
        minus_one_result = self.one_df.copy()*(-1)
        minus_one_result['Fantasy Points'] = self.sum_of_weights*(-1)

        pdtest.assert_frame_equal(stats_to_fantasy_points(minus_one_df),
                                  minus_one_result, check_dtype=False)
    
    def test_tensor_input_with_stat_index(self):
        pdtest.assert_frame_equal(stats_to_fantasy_points(self.one_tensor, stat_indices=self.stat_names),
                                  self.one_result, check_dtype=False)

    def test_tensor_input_with_stat_index_default_gives_right_result(self):
        pdtest.assert_frame_equal(stats_to_fantasy_points(self.one_tensor, stat_indices='default'),
                                  self.one_result, check_dtype=False)

    def test_tensor_input_without_stat_index_fails(self):
        with self.assertRaises(IndexError):
            stats_to_fantasy_points(self.one_tensor)

    def test_unnormalized_stats_with_norm_true_gives_wrong_result(self):
        with self.assertRaises(AssertionError):
            pdtest.assert_frame_equal(stats_to_fantasy_points(self.one_df, normalized=True),
                                      self.one_result, check_dtype=False)

    def test_normalized_stats_with_norm_false_gives_wrong_result(self):
        with self.assertRaises(AssertionError):
            pdtest.assert_frame_equal(stats_to_fantasy_points(self.one_df_normalized),
                                      self.one_result, check_dtype=False)

    def test_normalized_stats_with_norm_true_gives_right_result(self):
        pdtest.assert_frame_equal(stats_to_fantasy_points(self.one_df_normalized, normalized=True),
                                  self.one_result, check_dtype=False)

    def test_series_input_gives_right_result(self):
        # Custom setup
        one_series = self.one_df.iloc[0]

        pdtest.assert_frame_equal(stats_to_fantasy_points(one_series),
                                  self.one_result, check_dtype=False)

    def test_input_df_with_additional_column_names_gives_right_result(self):
        # Custom setup
        one_df_addl_column = self.one_df.copy()
        one_df_addl_column['XYZ'] = 1
        one_with_addl_column_result = one_df_addl_column.copy()
        one_with_addl_column_result['Fantasy Points'] = self.sum_of_weights

        pdtest.assert_frame_equal(stats_to_fantasy_points(one_df_addl_column),
                                  one_with_addl_column_result, check_dtype=False)

    def test_input_df_with_missing_stats_gives_error(self):
        # Custom setup
        stat_names_without_pass_yds = self.stat_names.copy()
        stat_names_without_pass_yds.remove('Pass Yds')
        sum_of_weights_without_pass_yds = sum([self.weights[stat] for stat in stat_names_without_pass_yds])
        one_df_without_pass_yds = self.one_df.copy()[stat_names_without_pass_yds]
        one_without_pass_yds_result = self.one_df.copy()[stat_names_without_pass_yds]
        one_without_pass_yds_result['Fantasy Points'] = sum_of_weights_without_pass_yds

        with self.assertRaises(KeyError):
            stats_to_fantasy_points(one_df_without_pass_yds)

    # Tear Down
    def tearDown(self):
        pass

class TestStatsToFantasyPoints_CustomWeights(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.weights = {'StatA':5,'StatB':-2,'StatC':0}
        self.thresholds = {'StatA':[-1, 1],
                           'StatB':[0,200],
                           'StatC':[-50,10000]}
        self.stat_names = list(self.weights.keys())
        self.sum_of_weights = sum(self.weights.values())
        # Common inputs for various test methods
        self.one_df = pd.DataFrame(data=[1]*len(self.stat_names), index=self.stat_names).T
        self.one_tensor = torch.tensor(self.one_df.values)
        self.one_df_normalized = normalize_stat(self.one_df, thresholds=self.thresholds)
        # Correct result for common inputs
        self.one_result = self.one_df.copy()
        self.one_result['Fantasy Points'] = self.sum_of_weights

    def test_zero_stats_equals_zero_points(self):
        # Custom setup
        zero_df = pd.DataFrame(data=[0]*len(self.stat_names), index=self.stat_names).T
        zero_result = zero_df.copy()
        zero_result['Fantasy Points'] = 0

        pdtest.assert_frame_equal(stats_to_fantasy_points(zero_df, scoring_weights=self.weights), 
                                  zero_result, check_dtype=False)

    def test_one_of_each_stat_equals_correct_points(self):
        pdtest.assert_frame_equal(stats_to_fantasy_points(self.one_df, scoring_weights=self.weights),
                                  self.one_result, check_dtype=False)

    def test_negative_one_of_each_stat_equals_correct_points(self):
        # Custom setup
        minus_one_df = self.one_df*(-1)
        minus_one_result = self.one_df.copy()*(-1)
        minus_one_result['Fantasy Points'] = self.sum_of_weights*(-1)

        pdtest.assert_frame_equal(stats_to_fantasy_points(minus_one_df, scoring_weights=self.weights),
                                  minus_one_result, check_dtype=False)
    
    def test_tensor_input_with_stat_index(self):
        pdtest.assert_frame_equal(stats_to_fantasy_points(self.one_tensor, stat_indices=self.stat_names, scoring_weights=self.weights),
                                  self.one_result, check_dtype=False)

    def test_tensor_input_with_stat_index_default_gives_error(self):
        with self.assertRaises(ValueError):
            stats_to_fantasy_points(self.one_tensor, stat_indices='default', scoring_weights=self.weights)

    def test_tensor_input_without_stat_index_fails(self):
        with self.assertRaises(IndexError):
            stats_to_fantasy_points(self.one_tensor, scoring_weights=self.weights)

    def test_unnormalized_stats_with_norm_true_gives_wrong_result(self):
        with self.assertRaises(AssertionError):
            pdtest.assert_frame_equal(stats_to_fantasy_points(self.one_df, normalized=True, norm_thresholds=self.thresholds, scoring_weights=self.weights),
                                      self.one_result, check_dtype=False)

    def test_normalized_stats_with_norm_false_gives_wrong_result(self):
        with self.assertRaises(AssertionError):
            pdtest.assert_frame_equal(stats_to_fantasy_points(self.one_df_normalized, norm_thresholds=self.thresholds, scoring_weights=self.weights),
                                      self.one_result, check_dtype=False)

    def test_normalized_stats_with_norm_true_gives_right_result(self):
        pdtest.assert_frame_equal(stats_to_fantasy_points(self.one_df_normalized, normalized=True, norm_thresholds=self.thresholds, scoring_weights=self.weights),
                                  self.one_result, check_dtype=False)

    def test_series_input_gives_right_result(self):
        # Custom setup
        one_series = self.one_df.iloc[0]

        pdtest.assert_frame_equal(stats_to_fantasy_points(one_series, scoring_weights=self.weights),
                                  self.one_result, check_dtype=False)

    def test_input_df_with_additional_column_names_gives_right_result(self):
        # Custom setup
        one_df_addl_column = self.one_df.copy()
        one_df_addl_column['XYZ'] = 1
        one_with_addl_column_result = one_df_addl_column.copy()
        one_with_addl_column_result['Fantasy Points'] = self.sum_of_weights

        pdtest.assert_frame_equal(stats_to_fantasy_points(one_df_addl_column, scoring_weights=self.weights),
                                  one_with_addl_column_result, check_dtype=False)

    def test_input_df_with_missing_stats_gives_error(self):
        # Custom setup
        stat_names_without_stat_a = self.stat_names.copy()
        stat_names_without_stat_a.remove('StatA')
        sum_of_weights_without_stat_a = sum([self.weights[stat] for stat in stat_names_without_stat_a])
        one_df_without_stat_a = self.one_df.copy()[stat_names_without_stat_a]
        one_without_stat_a_result = self.one_df.copy()[stat_names_without_stat_a]
        one_without_stat_a_result['Fantasy Points'] = sum_of_weights_without_stat_a

        with self.assertRaises(KeyError):
            stats_to_fantasy_points(one_df_without_stat_a, scoring_weights=self.weights)

    # Tear Down
    def tearDown(self):
        pass



if __name__ == '__main__':
    unittest.main()