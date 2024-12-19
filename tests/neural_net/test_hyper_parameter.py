import unittest
import random
import numpy as np
import pandas as pd
import pandas.testing as pdtest
import torch
# Class under test
from neural_net.hyper_parameter import HyperParameter
# Modules needed for test setup
import tests.utils_for_tests.mock_data as mock_data
import logging
import logging.config
from config.log_config import LOGGING_CONFIG

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')

class TestConstructor_HyperParameter(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.id_df = mock_data.id_df
        self.pbp_df = mock_data.pbp_df
        self.bs_df = mock_data.bs_df
        self.name = 'dataset'

    def test_basic_attributes_df_input(self):
        

    def test_basic_attributes_data_input(self):

    
    def test_missing_inputs_raises_error(self):


    def test_non_pandas_df_inputs_raises_error(self):


    def test_non_tensor_data_inputs_raises_error(self):


    def test_mixed_df_and_tensor_inputs_gives_correct_result(self):


    # Tear Down
    def tearDown(self):
        pass

