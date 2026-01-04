import torch
import torch.nn as nn

class MLPNetwork(nn.Module):
    """
    Standard Multi-Layer Perceptron to act as a baseline benchmark.
    Structure: Linear -> ReLU -> Linear -> ReLU
    """
    def __init__(self, layers_hidden):
        super().__init__()
        # layers_hidden example: [20, 64, 32]
        
        self.layers = nn.ModuleList()
        
        for i in range(len(layers_hidden) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(layers_hidden[i], layers_hidden[i+1]),
                nn.ReLU()
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x