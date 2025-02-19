"""Contains variables defining the configuration of Neural Networks used in Neural Net Fantasy Projections.

    Variables:
        nn_train_settings (dict): Settings to use in Neural Network training.
        default_nn_shape (dict): Neural Network layers and number of neurons per layer.
"""

# Settings to use in neural network training
nn_train_settings = {
    "max_epochs": 1,
    "n_epochs_to_stop": 5}

# Shape of neural network (can be reconfigured during object initialization)
default_nn_shape = {
    "players_input": 300,
    "teams_input": 32,
    "opps_input": 32,
    "stats_input": 29,
    "embedding_player": 50,
    "embedding_team": 10,
    "embedding_opp": 10,
    "linear_stack": 300,
    "stats_output": 12,
}
