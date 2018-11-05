import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size = 64, hidden_n = 1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.input = nn.Linear(state_size, hidden_size)
        self.input_relu = nn.ReLU()
        self.hidden_layers = nn.Sequential()
        for i in range(hidden_n):
            self.hidden_layers.add_module("hidden_"+str(i), nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.add_module("relu_"+str(i), nn.ReLU())
        self.output = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.input(state)
        x = self.input_relu(x)
        x = self.hidden_layers(x)
        x = self.output(x)
        return x
