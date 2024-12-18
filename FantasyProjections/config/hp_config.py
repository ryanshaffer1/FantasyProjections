"""Contains variables defining the configuration of hyper-parameters used in Neural Net Fantasy Projections.

    Variables: 
        hp_tuner_settings (dict): Defines settings ingested by the hyperparameter tuner.
        hp_defaults (dict): Defines Hyper-Parameters used by Neural Net, and default values for each.
"""

from torch import nn

# Hyper-Parameter Tuning Settings
hp_tuner_settings = {
    'optimize_hypers': True,
    'hyper_tuner_layers': 2,
    'hyper_tuner_steps_per_dim': 2,
    'scale_epochs_over_layers': True, # If True, max_epochs and n_epochs_to_stop will double with each layer of the hyperparameter grid search
    'plot_tuning_results': True,
}

# hp_defaults
# Each entry in the dict is of the form {hp_name: {hp_attributes}}.
# Attributes that can be defined for each hyper-parameter are listed in hyper_parameter.py.
# Any attributes not included in the dict will be set to their default values.
# Note that value MUST be defined here (otherwise there will be no "default" value for that hyper-parameter)
hp_defaults = {
    'mini_batch_size': {
        'value': 1000,
        'optimizable': False,
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
        # 'val_range': [100,900],
        # 'val_scale': 'linear'
    }
}
