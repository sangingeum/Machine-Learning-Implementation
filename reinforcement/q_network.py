from torch import nn


def create_fully_connected_layers(units_per_layer=[64, 32], use_relu=True):
    layers = list()
    layer_len = len(units_per_layer)
    if layer_len < 2:
        raise AssertionError
    for i in range(layer_len - 1):
        layers.append(nn.Linear(units_per_layer[i], units_per_layer[i + 1]))
        if use_relu and i < layer_len - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class basic_q_network(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer_units):
        super(basic_q_network, self).__init__()
        hidden_layer_units = [state_size, *hidden_layer_units, action_size]
        # Shared layers
        self.layers = create_fully_connected_layers(hidden_layer_units)
    def forward(self, state):
        return self.layers(state)

class dueling_q_network(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer_units, value_layer_units, advantage_layer_units):
        super(dueling_q_network, self).__init__()
        hidden_layer_units = [state_size, *hidden_layer_units]
        value_layer_units = [hidden_layer_units[-1], *value_layer_units, 1]
        advantage_layer_units = [hidden_layer_units[-1], *advantage_layer_units, action_size]
        # Shared layers
        self.shared_layers = create_fully_connected_layers(hidden_layer_units)
        # Value stream
        self.value_stream = create_fully_connected_layers(value_layer_units)
        # Advantage stream
        self.advantage_stream = create_fully_connected_layers(advantage_layer_units)

    def forward(self, state):
        x = self.shared_layers(state)
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        dim = len(advantages.size()) - 1
        q_values = values + (advantages - advantages.mean(dim=dim, keepdim=True))
        return q_values

