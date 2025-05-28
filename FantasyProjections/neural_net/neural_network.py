"""Creates and exports classes to be used in Neural Network creation, training, and evaluation.

    Classes:
        NeuralNet : PyTorch-based Neural Network object with a custom network architecture.
"""  # fmt: skip

import torch
from torch import nn


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
    def __init__(self, shape: dict):
        """Initializes a NeuralNetwork with a network architecture defined in FantasyProjections project docs.

            Args:
                shape (dict): number of neurons in each layer of the network, keyed by the names of each layer.

            Raises:
                ValueError: non-numeric value passed in shape.

        """  # fmt: skip

        super().__init__()
        self.shape = shape

        # Create embedding layers
        self.embedding = nn.ModuleDict()
        for embedding_name, embedding_size in shape.get("embedding", {}).items():
            input_size = shape["input"][embedding_name]
            embedding = nn.Sequential(
                nn.Linear(input_size, embedding_size, dtype=float),
                nn.ReLU(),
            )
            self.embedding[embedding_name] = embedding

        # Create linear stack layers
        self.linear_stack = nn.Sequential()
        prev_layer_size = shape["input"]["unembedded"] + sum(shape.get("embedding", {}).values())
        for layer_stack_size in shape["linear_stack"]:
            layer_input_size = prev_layer_size
            self.linear_stack += nn.Sequential(
                nn.Linear(layer_input_size, layer_stack_size, dtype=float),
                nn.ReLU(),
            )
            prev_layer_size = layer_stack_size

        # Create output layer
        self.output = nn.Sequential(
            nn.Linear(shape["linear_stack"][-1], shape["output"], dtype=float),
            nn.Sigmoid(),
        )

    # PUBLIC METHODS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the feedforward flow of information through the network, including the embedding and linear stack layers.

            Args:
                x (Tensor): input vector into Neural Net

            Returns:
                Tensor: output vector from Neural Net based on provided input

        """  # fmt: skip

        # Pass through all non-embedded inputs
        x_non_embedded = x[:, self.shape["input_indices"]["unembedded"]]

        # Compute all embeddings from inputs
        post_embeddings = []
        for e_name in self.embedding:
            x_embedded_input = x[:, self.shape["input_indices"][e_name]]
            x_post_embedding = self.embedding[e_name](x_embedded_input)
            post_embeddings.append(x_post_embedding)

        # Construct inputs vector to linear stack
        x_post_embeddings = torch.cat([x_non_embedded, *post_embeddings], dim=1)

        # Construct output
        x_post_stack = self.linear_stack(x_post_embeddings)
        logits = self.output(x_post_stack)
        return logits
