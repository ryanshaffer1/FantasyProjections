import logging
import logging.config
import random
import unittest

import numpy as np

# Modules needed for test setup
from config import hp_config
from config.log_config import LOGGING_CONFIG

# Class under test
from neural_net.hyper_parameter import HyperParameter

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")


class TestConstructor_HyperParameter(unittest.TestCase):
    # Set Up
    def setUp(self):
        pass

    def test_basic_attributes_no_optional_inputs_no_default(self):
        hp = HyperParameter(name="test")

        self.assertEqual(hp.name, "test")
        self.assertEqual(hp.optimizable, False)
        self.assertEqual(hp.value, 0)
        self.assertEqual(hp.val_range, [0])
        self.assertEqual(hp.val_scale, "none")
        self.assertEqual(hp.values, [0])

    def test_basic_attributes_no_optional_inputs_default_hp(self):
        defaults = hp_config.hp_defaults
        first_hp_name = next(iter(defaults.keys()))
        first_hp_settings = defaults[first_hp_name]
        first_hp_val = first_hp_settings.get("value", 0)

        hp = HyperParameter(name=first_hp_name)

        self.assertEqual(hp.name, first_hp_name)
        self.assertEqual(hp.value, first_hp_val)

    def test_basic_attributes_all_optional_inputs(self):
        name = "test"
        optimizable = True
        value = 50
        val_range = [1, 100]
        val_scale = "linear"

        hp = HyperParameter(name=name, optimizable=optimizable, value=value, val_range=val_range, val_scale=val_scale)

        self.assertEqual(hp.name, name)
        self.assertEqual(hp.optimizable, optimizable)
        self.assertEqual(hp.value, value)
        self.assertEqual(hp.val_range, val_range)
        self.assertEqual(hp.val_scale, val_scale)
        self.assertEqual(hp.values, [value])

    def test_missing_inputs_raises_error(self):
        with self.assertRaises(TypeError):
            HyperParameter()

    # Tear Down
    def tearDown(self):
        pass


class TestCopy_HyperParameter(unittest.TestCase):
    def setUp(self):
        self.hp = HyperParameter(name="test")

    def test_copy_produces_new_object(self):
        new_hp = self.hp.copy()

        self.assertNotEqual(id(self.hp), id(new_hp))

    def test_copy_produces_identical_object(self):
        new_hp = self.hp.copy()

        self.assertEqual(self.hp, new_hp)

    def tearDown(self):
        pass


class TestRandomizeInRange_HyperParameter(unittest.TestCase):
    def setUp(self):
        random.seed(0)

        self.hp_name = "test"
        self.settings = {
            "value": 50,
            "val_range": [1, 100],
        }

    def test_linear_hp_randomizes_in_range(self):
        self.settings["val_scale"] = "linear"

        hp = HyperParameter(name=self.hp_name, optimizable=True, **self.settings)
        vals = hp.randomize_in_range(1000)

        self.assertTrue(all(np.array(vals) <= max(hp.val_range)))
        self.assertTrue(all(np.array(vals) >= min(hp.val_range)))

    def test_log_hp_randomizes_in_range(self):
        self.settings["val_scale"] = "log"

        hp = HyperParameter(name=self.hp_name, optimizable=True, **self.settings)
        vals = hp.randomize_in_range(1000)

        self.assertTrue(all(np.array(vals) <= max(hp.val_range)))
        self.assertTrue(all(np.array(vals) >= min(hp.val_range)))

    def test_selection_hp_randomizes_in_range(self):
        self.settings["val_scale"] = "selection"
        self.settings["val_range"] = ["a", "b"]

        hp = HyperParameter(name=self.hp_name, optimizable=True, **self.settings)
        vals = hp.randomize_in_range(1000)

        self.assertTrue(all(val in hp.val_range for val in vals))

    def test_none_val_scale_hp_returns_hp_value_repeated(self):
        self.settings["val_scale"] = "none"
        hp = HyperParameter(name=self.hp_name, optimizable=True, **self.settings)
        vals = hp.randomize_in_range(1000)

        self.assertEqual(vals, [hp.value] * 1000)

    def test_other_val_scale_hp_returns_hp_value_repeated(self):
        self.settings["val_scale"] = "xyz"
        hp = HyperParameter(name=self.hp_name, optimizable=True, **self.settings)
        vals = hp.randomize_in_range(1000)

        self.assertEqual(vals, [hp.value] * 1000)

    def test_n_equals_one_returns_list_with_length_one(self):
        hp = HyperParameter(name=self.hp_name, optimizable=True, **self.settings)
        vals = hp.randomize_in_range(1)

        self.assertEqual(len(vals), 1)

    def test_n_equals_ten_returns_list_with_length_ten(self):
        hp = HyperParameter(name=self.hp_name, optimizable=True, **self.settings)
        vals = hp.randomize_in_range(10)

        self.assertEqual(len(vals), 10)

    def test_missing_n_returns_error(self):
        hp = HyperParameter(name=self.hp_name, optimizable=True, **self.settings)

        with self.assertRaises(TypeError):
            hp.randomize_in_range()

    def tearDown(self):
        pass


class TestAdjustRange_HyperParameter(unittest.TestCase):
    def setUp(self):
        random.seed(0)

        self.hp_name = "test"
        self.settings = {
            "value": 50,
            "val_scale": "linear",
            "val_range": [0, 100],
        }

    def test_linear_hp_recenters_and_rescales_correctly(self):
        hp = HyperParameter(name=self.hp_name, optimizable=True, **self.settings)
        new_range = hp.adjust_range(center_point=40, scale_factor=0.25)

        self.assertEqual(new_range, [40 - 12.5, 40 + 12.5])

    def test_log_hp_recenters_and_rescales_correctly(self):
        self.settings["val_scale"] = "log"
        self.settings["val_range"] = [1, 100]
        hp = HyperParameter(name=self.hp_name, optimizable=True, **self.settings)
        new_range = hp.adjust_range(center_point=10, scale_factor=0.5)

        self.assertEqual(new_range, [np.sqrt(10), 10**1.5])

    def test_selection_hp_returns_center_value_provided(self):
        hp = HyperParameter(name=self.hp_name, optimizable=True, val_scale="selection", val_range=["a", "b"], value="b")
        new_range = hp.adjust_range(center_point="b")

        self.assertEqual(new_range, ["b"])

    def test_no_scale_factor_defaults_to_one(self):
        hp = HyperParameter(name=self.hp_name, optimizable=True, **self.settings)
        new_range = hp.adjust_range(center_point=50)

        self.assertEqual(new_range, [0, 100])

    def test_exceed_boundary_true_moves_range_beyond_original_boundary(self):
        hp = HyperParameter(name=self.hp_name, optimizable=True, **self.settings)
        new_range = hp.adjust_range(center_point=90, scale_factor=0.5, exceed_boundary=True)
        expected_range = [90 - 25, 90 + 25]

        self.assertEqual(new_range, expected_range)

    def test_exceed_boundary_false_clips_range_to_original_boundary(self):
        hp = HyperParameter(name=self.hp_name, optimizable=True, **self.settings)
        new_range = hp.adjust_range(center_point=90, scale_factor=0.5, exceed_boundary=False)
        expected_range = [50, 100]

        self.assertEqual(new_range, expected_range)

    def tearDown(self):
        pass
