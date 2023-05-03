from torch import nn
import torch
import math
class graph_convolutional_layer(nn.Module):
    def __init__(self, input_size, output_size, use_bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(size=(input_size, output_size)))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(output_size, )))
        self.use_bias = use_bias
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X, adj):
        X = X @ self.weight
        X = torch.sparse.mm(adj, X)
        if self.use_bias:
            X += self.bias
        return X



