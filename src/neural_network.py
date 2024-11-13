import torch
from torch import nn

# Info on input data structure (horrible variable struct, and this should
# be passed from a special file or from preProcessNNData.py):


def index_range(prev_range, length):
    return range(max(prev_range) + 1, max(prev_range) + 1 + length)


N_STATS = 25
N_POSITIONS = 4
N_PLAYERS = 300
N_TEAMS = 32
N_OPPONENTS = 32
stats_inds = range(0, N_STATS)
position_inds = index_range(stats_inds, N_POSITIONS)
players_inds = index_range(position_inds, N_PLAYERS)
teams_inds = index_range(players_inds, N_TEAMS)
opponents_inds = index_range(teams_inds, N_OPPONENTS)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_player = nn.Sequential(
            nn.Linear(N_PLAYERS, 50, dtype=float),
            nn.ReLU()
        )
        self.embedding_team = nn.Sequential(
            nn.Linear(N_TEAMS, 10, dtype=float),
            nn.ReLU()
        )
        self.embedding_opp = nn.Sequential(
            nn.Linear(N_OPPONENTS, 10, dtype=float),
            nn.ReLU()
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(N_STATS + N_POSITIONS + 50 + 10 + 10, 300, dtype=float),
            nn.ReLU(),
            nn.Linear(300, 12, dtype=float),
            nn.Sigmoid()
        )

    def forward(self, x):
        player_embedding = self.embedding_player(x[:, players_inds])
        team_embedding = self.embedding_team(x[:, teams_inds])
        opp_embedding = self.embedding_opp(x[:, opponents_inds])
        x_embedded = torch.cat((x[:,
                                  0:max(position_inds) + 1],
                                player_embedding,
                                team_embedding,
                                opp_embedding),
                               dim=1)
        logits = self.linear_stack(x_embedded)
        return logits
