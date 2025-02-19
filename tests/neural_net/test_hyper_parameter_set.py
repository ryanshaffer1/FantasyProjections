import logging
import logging.config
import unittest

from config.log_config import LOGGING_CONFIG

# Modules needed for test setup
from neural_net.hyper_parameter import HyperParameter

# Class under test
from neural_net.hyper_parameter_set import HyperParameterSet

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")


class TestConstructor_HyperParameterSet(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.hp1 = HyperParameter(name="hp1", value=4, optimizable=False)
        self.hp2 = HyperParameter(name="hp2", value=-1, optimizable=True, val_range=[-2, -0.5], val_scale="linear")
        self.hp_dict = {
            "hp1": {"value": 4, "optimizable": False},
            "hp2": {"value": -1, "optimizable": True, "val_range": [-2, -0.5], "val_scale": "linear"},
        }
        self.hp_dict_extra = self.hp_dict.copy()
        self.hp_dict_extra["hp3"] = {"value": 1000, "optimizable": False}

    def test_input_tuple_of_hyperparameters_works(self):
        hpset = HyperParameterSet(hp_set=(self.hp1, self.hp2))

        self.assertEqual(hpset.hyper_parameters, (self.hp1, self.hp2))

    def test_input_dict_of_hp_settings_works(self):
        hpset = HyperParameterSet(hp_dict=self.hp_dict)

        self.assertEqual(hpset.hyper_parameters, (self.hp1, self.hp2))

    def test_input_both_tuple_and_dict_favors_tuple(self):
        hpset = HyperParameterSet(hp_set=(self.hp1, self.hp2), hp_dict=self.hp_dict_extra)

        self.assertEqual(hpset.hyper_parameters, (self.hp1, self.hp2))

    def test_input_single_hyperparameter_no_tuple_works(self):
        hpset = HyperParameterSet(hp_set=self.hp1)

        self.assertEqual(hpset.hyper_parameters, (self.hp1,))

    def test_input_single_hyperparameter_in_tuple_works(self):
        hpset = HyperParameterSet(hp_set=(self.hp1,))

        self.assertEqual(hpset.hyper_parameters, (self.hp1,))

    def test_missing_inputs_raises_error(self):
        with self.assertRaises(ValueError):
            HyperParameterSet()

    # Tear Down
    def tearDown(self):
        pass


class TestGet_HyperParameterSet(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.hp1 = HyperParameter(name="hp1", value=4, optimizable=False)
        self.hp2 = HyperParameter(name="hp2", value=-1, optimizable=True, val_range=[-2, -0.5], val_scale="linear")
        self.hp_set = HyperParameterSet(hp_set=(self.hp1, self.hp2))

    def test_get_first_hp(self):
        hp = self.hp_set.get("hp1")

        self.assertEqual(hp, self.hp1)

    def test_get_second_hp(self):
        hp = self.hp_set.get("hp2")

        self.assertEqual(hp, self.hp2)

    def test_get_hp_not_in_set_raises_error(self):
        with self.assertRaises(ValueError):
            self.hp_set.get("hp3")

    def test_get_with_index_not_name_raises_error(self):
        with self.assertRaises(ValueError):
            self.hp_set.get(0)

    # Tear Down
    def tearDown(self):
        pass


class TestSetValues_HyperParameterSet(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.values_hp1 = [4, 5, 6, 7]
        self.values_hp2 = [-3, -2, -1, -0.5]

        self.hp1 = HyperParameter(name="hp1", value=4, optimizable=False)
        self.hp1.values = self.values_hp1
        self.hp2 = HyperParameter(name="hp2", value=-1, optimizable=True, val_range=[-2, -0.5], val_scale="linear")
        self.hp2.values = self.values_hp2

        self.hp_set = HyperParameterSet(hp_set=(self.hp1, self.hp2))

    def test_set_first_value(self):
        ind = 0
        self.hp_set.set_values(ind)

        self.assertEqual(self.hp_set.hyper_parameters[0].value, self.values_hp1[ind])
        self.assertEqual(self.hp_set.hyper_parameters[1].value, self.values_hp2[ind])

    def test_set_second_value(self):
        ind = 1
        self.hp_set.set_values(ind)

        self.assertEqual(self.hp_set.hyper_parameters[0].value, self.values_hp1[ind])
        self.assertEqual(self.hp_set.hyper_parameters[1].value, self.values_hp2[ind])

    def test_set_value_outside_range_raises_error(self):
        with self.assertRaises(IndexError):
            self.hp_set.set_values(5)

    def test_input_non_int_index_raises_error(self):
        with self.assertRaises(TypeError):
            self.hp_set.set_values([0])

    # Tear Down
    def tearDown(self):
        pass


class TestToDict_HyperParameterSet(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.hp1 = HyperParameter(name="hp1", value=4, optimizable=False)
        self.hp2 = HyperParameter(name="hp2", value=-1, optimizable=True, val_range=[-2, -0.5], val_scale="linear")

        self.hp_set = HyperParameterSet(hp_set=(self.hp1, self.hp2))

        self.expected_dict = {"hp1": 4, "hp2": -1}

    def test_dict_matches_expected_return(self):
        hp_dict = self.hp_set.to_dict()

        self.assertEqual(hp_dict, self.expected_dict)

    # Tear Down
    def tearDown(self):
        pass
