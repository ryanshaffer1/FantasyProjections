"""Contains variables defining the configuration of hyper-parameters used in Neural Net Fantasy Projections.

    Variables: 
        hp_tuner_settings (dict): Defines settings ingested by the hyperparameter tuner.
        hp_defaults (dict): Defines Hyper-Parameters used by Neural Net, and default values for each.
"""

from torch import nn

# Hyper-Parameter Tuning Settings
hp_tuner_settings = {
    'optimize_hypers': False,
    'hyper_tuner_layers': 2,
    'hyper_tuner_steps_per_dim': 2,
    'plot_tuning_results': True,
}
# hp_tuner_settings = {
#     'optimize_hypers': True,
#     'max_samples': 200,
#     'r_percentile': 0.1,
#     'v_expect_imp': 0.1,
#     'plot_tuning_results': True,
# }

# hp_defaults
# Each entry in the dict is of the form {hp_name: {hp_attributes}}.
# Attributes that can be defined for each hyper-parameter are listed in hyper_parameter.py.
# Any attributes not included in the dict will be set to their default values.
# Note that value MUST be defined here (otherwise there will be no "default" value for that hyper-parameter)
hp_defaults = {
    'mini_batch_size': {
        'value': 1000,
        'optimizable': False,
        'val_range': [100, 10000],
        'val_scale': 'log'
    },
    'learning_rate': {
        'value': 50,
        'optimizable': True,
        'val_range': [1e-2,100],
        'val_scale': 'log',
    },
    'lmbda': {
        'value': 0,
        'optimizable': True,
        'val_range': [1e-7,1e-3],
        'val_scale': 'log',
    },
    'loss_fn': {
        'value': nn.MSELoss(),
        'optimizable': False
    },
    'linear_stack': {
        'value': 300,
        'optimizable': False,
        'val_range': [200,800],
        'val_scale': 'linear'
    }
}
