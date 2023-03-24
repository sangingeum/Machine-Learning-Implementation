import torch
from torch import nn

class neural_network_regression(nn.Module):
    def __init__(self, units_per_layer=[2, 5, 1]):
        super().__init__()
        if len(units_per_layer) < 2:
            print("Units_per_layer should be longer than 2.")
            units_per_layer = [2, 5, 1]
        self.layers = nn.ModuleList([nn.Linear(units_per_layer[i], units_per_layer[i+1], dtype=torch.float32) for i in range(len(units_per_layer) - 1)])
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
            y = self.relu(y)
        return y





