import torch
from torch import nn

# Info on input data structure (horrible variable struct, and this should
# be passed from a special file or from preProcessNNData.py):


def index_range(prev_range, length):
    return range(max(prev_range) + 1, max(prev_range) + 1 + length)

# Shape of neural network (can be reconfigured during object initialization)
default_shape = {
    'players_input': 300,
    'teams_input': 32,
    'stats_input': 25,
    'positions_input': 4,
    'embedding_player': 50,
    'embedding_team': 10,
    'embedding_opp': 10,
    'linear_stack': 300,
    'stats_output': 12,
}
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self,shape=None):
        super().__init__()

        # Establish shape based on optional inputs
        if not shape:
            shape = default_shape
        else:
            for (key,val) in default_shape.items():
                if key not in shape:
                    shape[key] = val
        # Indices of input vector that correspond to each "category" (needed bc they are embedded separately)
        self.stats_inds = range(0, shape['stats_input'])
        self.position_inds = index_range(self.stats_inds, shape['positions_input'])
        self.players_inds = index_range(self.position_inds, shape['players_input'])
        self.teams_inds = index_range(self.players_inds, shape['teams_input'])
        self.opponents_inds = index_range(self.teams_inds, shape['teams_input'])

        self.embedding_player = nn.Sequential(
            nn.Linear(shape['players_input'], shape['embedding_player'], dtype=float),
            nn.ReLU()
        )
        self.embedding_team = nn.Sequential(
            nn.Linear(shape['teams_input'], shape['embedding_team'], dtype=float),
            nn.ReLU()
        )
        self.embedding_opp = nn.Sequential(
            nn.Linear(shape['teams_input'], shape['embedding_opp'], dtype=float),
            nn.ReLU()
        )
        n_input_to_linear_stack = sum(shape[item] for item in ['stats_input','positions_input','embedding_player','embedding_team','embedding_opp'])
        self.linear_stack = nn.Sequential(
            nn.Linear(n_input_to_linear_stack, shape['linear_stack'], dtype=float),
            nn.ReLU(),
            nn.Linear(shape['linear_stack'], shape['stats_output'], dtype=float),
            nn.Sigmoid()
        )

    def forward(self, x):
        player_embedding = self.embedding_player(x[:, self.players_inds])
        team_embedding = self.embedding_team(x[:, self.teams_inds])
        opp_embedding = self.embedding_opp(x[:, self.opponents_inds])
        x_embedded = torch.cat((x[:,
                                  0:max(self.position_inds) + 1],
                                player_embedding,
                                team_embedding,
                                opp_embedding),
                               dim=1)
        logits = self.linear_stack(x_embedded)
        return logits
