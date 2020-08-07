import os
import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, fc1_dims,
                 fc2_dims, alpha, name, checkpoint_dir):
        super().__init__()
        self.name = name
        self.filename = os.path.join(checkpoint_dir, name)

        self.fc1 = nn.Linear(*input_shape, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)

        self.optim = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        fc1 = F.relu(self.fc1(state))
        fc2 = F.relu(self.fc2(fc1))
        mu = T.tanh(self.mu(fc2))
        return mu

    def save_checkpoint(self):
        print(f'... saving checkpoint as {self.name} ...')
        T.save(self.state_dict(), self.filename)

    def load_checkpoint(self):
        print(f'... loading checkpoint from {self.name} ...')
        self.load_state_dict(T.load(self.filename))

