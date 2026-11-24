import random

import numpy as np
import torch
import torch.nn as nn

from src.dqn import DQN, DuelingDQN


class DDQNTrainer:

    def __init__(
        self,
        observation_size=48,
        n_actions=5,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        double=True,
        dueling=False
    ):

        self.double = double
        net = DuelingDQN if dueling else DQN

        self.device = torch.device("cpu")

        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

        self.training_steps = 0
        self.target_update = 200

        # Réseau principal
        self.online = net(
            observation_size,
            n_actions
        ).to(self.device)

        # Réseau cible
        self.target = net(
            observation_size,
            n_actions
        ).to(self.device)

        self.target.load_state_dict(
            self.online.state_dict()
        )

        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=learning_rate
        )

        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state):

        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        # Exploitation
        with torch.no_grad():

            state = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            return self.online(state).argmax(dim=1).item()

    def train_step(self, buffer):

        if len(buffer) < max(self.batch_size, 500):
            return None

        batch = buffer.sample(self.batch_size)

        states = torch.as_tensor(
            np.stack([t.state for t in batch]),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            [t.action for t in batch],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            [t.reward for t in batch],
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.as_tensor(
            np.stack([t.next_state for t in batch]),
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            [t.done for t in batch],
            dtype=torch.float32,
            device=self.device
        )

        # Q-values actuelles
        q_values = self.online(states).gather(
            1, actions
        ).squeeze(1)

        # Cible Double DQN
        with torch.no_grad():

            selector = self.online if self.double else self.target

            next_actions = selector(
                next_states
            ).argmax(dim=1, keepdim=True)

            next_q_values = self.target(
                next_states
            ).gather(
                1, next_actions
            ).squeeze(1)

            targets = (
                rewards
                + self.gamma * next_q_values * (1 - dones)
            )

        # Optimisation
        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        self.training_steps += 1

        # Mise à jour du réseau cible
        if self.training_steps % self.target_update == 0:

            self.target.load_state_dict(
                self.online.state_dict()
            )

        # Réduction progressive de l'exploration
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

        return loss.item()

    def save(self, path):

        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps
        }, path)