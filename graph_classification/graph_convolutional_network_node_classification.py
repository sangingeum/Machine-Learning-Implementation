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
        """

        #self.GCN_layers = self._build_GCN_layers(input_size=input_size, GCN_units=GCN_units, use_bias=use_bias, drop_out=drop_out)
        #self.MLP_layers = self._build_MLP_layers(output_size=output_size, GCN_units=GCN_units,mlp_units=mlp_units, use_bias=use_bias, drop_out=drop_out)

                for layer in self.GCN_layers:
            X = nn.functional.relu(layer(X, adj))
        for layer in self.MLP_layers:
            X = layer(X)


        return X
    def _build_GCN_layers(self, input_size, GCN_units, use_bias, drop_out):
        GCN_layers = nn.ModuleList()
        GCN_units = [input_size, *GCN_units]
        for i in range(len(GCN_units) - 1):
            GCN_layers.append(graph_convolutional_layer(input_size=GCN_units[i],
                                                        output_size=GCN_units[i + 1],
                                                        use_bias=use_bias))
            if i < len(GCN_units) - 2:
                if drop_out > 0:
                    GCN_layers.append(nn.Dropout(p=drop_out))
        return nn.Sequential(*GCN_layers)

    def _build_MLP_layers(self, output_size, GCN_units, mlp_units, use_bias, drop_out):
        MLP_layers = nn.ModuleList()
        mlp_units = [GCN_units[-1], *mlp_units, output_size]
        for i in range(len(mlp_units) - 1):
            MLP_layers.append(nn.Linear(mlp_units[i], mlp_units[i + 1], bias=use_bias))
            if i < len(mlp_units) - 2:
                if drop_out > 0:
                    MLP_layers.append(nn.Dropout(p=drop_out))
                MLP_layers.append(nn.ReLU())
        MLP_layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*MLP_layers)

"""