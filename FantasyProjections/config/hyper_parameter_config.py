"""Contains variables defining the configuration of hyper-parameters used in Neural Net Fantasy Projections.

    Variables: 
        hp_tuner_settings (dict): Defines settings ingested by the hyperparameter tuner.
        mini_batch_size (HyperParameter): size of training data batches used in SGD training iterations
        learning_rate (HyperParameter): Learning Rate used in SGD optimizer
        lmbda (HyperParameter): L2 Regularization Parameter used in SGD optimizer
        loss_fn (HyperParameter): Equation to use to calculate loss from ideal solution (e.g. Mean Squared Error (MSE))
"""

from torch import nn
from tuners.hyper_tuner import HyperParameter, HyperParameterSet

# Hyper-Parameter Tuning Settings
hp_tuner_settings = {
    'optimize_hypers': False,
    'hyper_tuner_layers': 2,
    'hyper_tuner_steps_per_dim': 2,
    'scale_epochs_over_layers': True, # If True, max_epochs and n_epochs_to_stop will double with each layer of the hyperparameter grid search
    'plot_tuning_results': True,
}

# Hyper-parameters (automatically tuned if optimize_hypers = True)
mini_batch_size = HyperParameter('mini_batch_size',
                                optimizable=False,
                                value=1000,
                                #  val_range = [1000,10000],
                                #  val_range=[100,10000],
                                #  val_scale='log',
                                #  num_steps=HYPER_TUNER_STEPS_PER_DIM
                                )
learning_rate = HyperParameter('learning_rate',
                            optimizable=True,
                            value=50,
                            # val_range=[5,50],
                            val_range=[1e-2,100],
                            val_scale='log',
                            num_steps=hp_tuner_settings['hyper_tuner_steps_per_dim'])
lmbda = HyperParameter('lmbda',
                    optimizable=True,
                    value=0,
                    # val_range=[1e-7,1e-5],
                    val_range=[1e-7,1e-3],
                    val_scale='log',
                    num_steps=hp_tuner_settings['hyper_tuner_steps_per_dim'])
loss_fn = HyperParameter('loss_fn',
                        optimizable=False,
                        value=nn.MSELoss()
                        # if optimizable, add kwargs: ,val_range=[nn.MSELoss(),nn.CrossEntropyLoss()],val_scale='selection')
                        )

# Set of all hyper-parameters (collected so that they can be varied/optimized together)
param_set = HyperParameterSet((mini_batch_size,learning_rate,lmbda,loss_fn),optimize=hp_tuner_settings['optimize_hypers'])
