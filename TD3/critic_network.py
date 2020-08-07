import os
import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, fc1_dims,
                 fc2_dims, beta, name, checkpoint_dir):
        super().__init__()
        self.name = name
        self.filename = os.path.join(checkpoint_dir, name)
        
        self.fc1 = nn.Linear(input_shape[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        
        self.optim = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state, action):
        state_action = T.cat((state, action), dim=1)
        fc1 = F.relu(self.fc1(state_action))
        fc2 = F.relu(self.fc2(fc1))
        q = self.q(fc2)
        return q
    
    def save_checkpoint(self):
        print(f'... saving checkpoint as {self.name} ...')
        T.save(self.state_dict(), self.filename)
        
    def load_checkpoint(self):
        print(f'... loading checkpoint from {self.name} ...')
        self.load_state_dict(T.load(self.filename))
