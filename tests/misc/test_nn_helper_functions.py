import unittest
import pandas as pd
import pandas.testing as pdtest
from misc.nn_helper_functions import normalize_stat
from config import stats_config

# Set up same logger as project code
import logging
import logging.config
from config.log_config import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')



class TestNormalizeStat(unittest.TestCase):

    # Set Up
    def setUp(self):
        self.thresholds = stats_config.default_norm_thresholds
        self.threshold_mins = [x[0] for x in self.thresholds.values()]
        self.threshold_maxs = [x[1] for x in self.thresholds.values()]
        # self.data = [0,0,0,0,0,0,0,0,0,0,0,0]
        # self.col = pd.Series(self.data,index=range(len(self.data)))
        self.column_names = list(self.thresholds.keys())

    def test_within_thresholds(self):
        self.assertEqual(1,1)

    def test_beyond_thresholds(self):
        self.assertEqual(1,1)


    # Tests of normalization calculations across input ranges
    def test_min_thresholds_normalize_to_zero(self):
        # for data in 
        result = normalize_stat(pd.Series(data=self.threshold_mins, index=self.column_names), 
                       thresholds=self.thresholds)
        all_zeros_series = pd.Series(data=[0]*len(self.column_names), index=self.column_names)
        pdtest.assert_series_equal(result,all_zeros_series)
        # self.assertEqual(result,all_zeros_series)

    def test_below_min_thresholds_normalize_to_zero(self):
        pass

    def test_max_thresholds_normalize_to_one(self):
        pass

    def test_above_min_thresholds_normalize_to_one(self):
        pass

    def test_midway_values_normalize_to_one_half(self):
        pass

    # Tear Down
    def tearDown(self):
        pass
    

if __name__ == '__main__':
    unittest.main()