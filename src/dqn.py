import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, observation_size=48, n_actions=5):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, n_actions)
        )

    def forward(self, x):

        return self.network(x)


class DuelingDQN(nn.Module):

    def __init__(self, observation_size=48, n_actions=5):

        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, n_actions)

    def forward(self, x):

        h = self.features(x)
        a = self.advantage(h)

        return self.value(h) + a - a.mean(dim=1, keepdim=True)