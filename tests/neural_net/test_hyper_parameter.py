import unittest
# Class under test
from neural_net.hyper_parameter import HyperParameter
# Modules needed for test setup
from config import hp_config
import logging
import logging.config
from config.log_config import LOGGING_CONFIG

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')

class TestConstructor_HyperParameter(unittest.TestCase):
    # Set Up
    def setUp(self):
        pass

    def test_basic_attributes_no_optional_inputs_no_default(self):
        hp = HyperParameter(name='test')
        
        self.assertEqual(hp.name, 'test')
        self.assertEqual(hp.optimizable, False)
        self.assertEqual(hp.value, 0)
        self.assertEqual(hp.val_range, [0])
        self.assertEqual(hp.val_scale, 'none')
        self.assertEqual(hp.values, [0])

    def test_basic_attributes_no_optional_inputs_default_hp(self):
        defaults = hp_config.hp_defaults
        first_hp_name = list(defaults.keys())[0]
        first_hp_settings = defaults[first_hp_name]
        first_hp_val = first_hp_settings.get('value', 0)

        hp = HyperParameter(name=first_hp_name)
        
        self.assertEqual(hp.name, first_hp_name)
        self.assertEqual(hp.optimizable, first_hp_settings.get('optimizable',False))
        self.assertEqual(hp.value, first_hp_val)
        self.assertEqual(hp.val_range, first_hp_settings.get('val_range',[first_hp_val]))
        self.assertEqual(hp.val_scale, first_hp_settings.get('val_scale','none'))
        self.assertEqual(hp.values, first_hp_settings.get('val_range',[first_hp_val]))
        

    def test_basic_attributes_all_optional_inputs(self):
        name='test'
        optimizable=True
        value=50
        val_range = [1, 100]
        val_scale='linear'

        hp = HyperParameter(name=name, 
                            optimizable=optimizable,
                            value=value,
                            val_range=val_range,
                            val_scale=val_scale)
        
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

