"""Creates and exports classes to be used in Neural Network creation, training, and evaluation.

    Classes:
        NeuralNet : PyTorch-based Neural Network object with a custom network architecture.
"""  # fmt: skip

import torch
from torch import nn

from config.nn_config import default_nn_shape


class NeuralNetwork(nn.Module):
    """Neural Network model implemented using PyTorch.nn, tailored to the data structures and needs of fantasy_projections.

        Attributes:
            stats_inds (list): lazy, list of indices in input vector corresponding to "raw game stats"
            players_inds (list): lazy, list of indices in input vector corresponding to player IDs
            teams_inds (list): lazy, list of indices in input vector corresponding to team IDs
            opponents_inds (list): lazy, list of indices in input vector corresponding to opponent IDs
            embedding_player (nn.Sequential): network layer embedding the player ID information. Parallel to other embedding layers.
            embedding_team (nn.Sequential): network layer embedding the team ID information. Parallel to other embedding layers.
            embedding_opp (nn.Sequential): network layer embedding the opponent ID information. Parallel to other embedding layers.
            linear_stack (nn.Sequential): network layer(s) containing interconnections between embeddings/other stats and output layer.

        Public Methods:
            forward : Defines the feedforward flow of information through the network, including the embedding and linear stack layers.

    """  # fmt: skip

    # CONSTRUCTOR
    def __init__(self, shape=None):
        """Initializes a NeuralNetwork with a network architecture defined in FantasyProjections project docs.

            Args:
                shape (dict, optional): number of neurons in each layer of the network, keyed by the names of each layer.
                Defaults to dict default_shape.

            Raises:
                ValueError: non-numeric value passed in shape.

        """  # fmt: skip

        super().__init__()

        # Establish shape based on optional inputs
        if not shape:
            shape = default_nn_shape
        else:
            for key, val in default_nn_shape.items():
                if key not in shape:
                    shape[key] = val

        # Ensure all values are of type int
        try:
            shape = {key: int(val) for key, val in shape.items()}
        except ValueError as e:
            msg = "Neural Net input with a non-numeric value in shape"
            raise ValueError(msg) from e

        # Indices of input vector that correspond to each "category" (needed bc they are embedded separately)
        self.stats_inds = range(shape["stats_input"])
        self.players_inds = self.__index_range(self.stats_inds, shape["players_input"])
        self.teams_inds = self.__index_range(self.players_inds, shape["teams_input"])
        self.opponents_inds = self.__index_range(self.teams_inds, shape["opps_input"])

        self.embedding_player = nn.Sequential(
            nn.Linear(shape["players_input"], shape["embedding_player"], dtype=float),
            nn.ReLU(),
        )
        self.embedding_team = nn.Sequential(
            nn.Linear(shape["teams_input"], shape["embedding_team"], dtype=float),
            nn.ReLU(),
        )
        self.embedding_opp = nn.Sequential(
            nn.Linear(shape["opps_input"], shape["embedding_opp"], dtype=float),
            nn.ReLU(),
        )
        n_input_to_linear_stack = sum(
            shape[item] for item in ["stats_input", "embedding_player", "embedding_team", "embedding_opp"]
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(n_input_to_linear_stack, shape["linear_stack"], dtype=float),
            nn.ReLU(),
            nn.Linear(shape["linear_stack"], shape["stats_output"], dtype=float),
            nn.Sigmoid(),
        )

    # PUBLIC METHODS

    def forward(self, x):
        """Defines the feedforward flow of information through the network, including the embedding and linear stack layers.

            Args:
                x (tensor): input vector into Neural Net

            Returns:
                tensor: output vector from Neural Net based on provided input

        """  # fmt: skip

        player_embedding = self.embedding_player(x[:, self.players_inds])
        team_embedding = self.embedding_team(x[:, self.teams_inds])
        opp_embedding = self.embedding_opp(x[:, self.opponents_inds])
        x_embedded = torch.cat((x[:, 0 : max(self.stats_inds) + 1], player_embedding, team_embedding, opp_embedding), dim=1)
        logits = self.linear_stack(x_embedded)
        return logits

    # PRIVATE METHODS

    def __index_range(self, prev_range, length):
        # Returns the next range of length "length", starting from the end of a previous range "prev_range"
        return range(max(prev_range) + 1, max(prev_range) + 1 + length)
