from graph_classification.graph_convolutional_layer import *

class graph_convolutional_network_node_classification(nn.Module):
    def __init__(self, input_size, output_size, use_bias=True, drop_out=0.0):
        super().__init__()
        self.drop_out = drop_out
        self.gl1 = graph_convolutional_layer(input_size=input_size, output_size=16, use_bias=use_bias)
        self.gl2 = graph_convolutional_layer(input_size=16, output_size=output_size, use_bias=use_bias)

    def forward(self, X, adj):
        X = self.gl1(X, adj)
        X = nn.functional.dropout(X, p=self.drop_out)
        X = nn.functional.relu(X)
        X = self.gl2(X, adj)
        X = nn.functional.softmax(X, dim=1)
        return X
   