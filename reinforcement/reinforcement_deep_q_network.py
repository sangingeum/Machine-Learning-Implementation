from torch import nn
from reinforcement.replay_buffer import *
from copy import deepcopy
from reinforcement.prioritized_replay_buffer import *
class reinforcement_deep_q_network:
    """
    n-step DQN with a replay buffer
    """
    def __init__(self, units_per_layer=[4, 64, 128, 64, 2], use_PER=False):
        # model creation
        self.network = self._build_model(units_per_layer)
        # hyper parameter setting
        self.state_size = units_per_layer[0]
        self.action_size = units_per_layer[-1]
        self.curr_step = 0
        self.learning_rate = 0.0003
        self.buffer_size = 50000
        self.learning_start_buffer_size = 25000
        self.batch_size = 128
        self.epsilon = 1.0
        self.min_epsilon = 0.005
        self.epsilon_decay = 0.9999
        self.gamma = 0.85
        self.n_step = 1
        self.gradient_update_freq = 1
        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.HuberLoss()
        self.use_PER = use_PER
        if use_PER:
            self.replay_buffer = prioritized_replay_buffer(buffer_size=self.buffer_size, n_step=self.n_step, gamma=self.gamma)
        else:
            self.replay_buffer = replay_buffer(buffer_size=self.buffer_size, n_step=self.n_step, gamma=self.gamma)
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = self.network.to(self.device)

    def _build_model(self, units_per_layer):
        layers = list()
        layer_len = len(units_per_layer)
        if layer_len < 2:
            raise AssertionError
        for i in range(layer_len - 1):
            layers.append(nn.Linear(units_per_layer[i], units_per_layer[i + 1]))
            if i < layer_len - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def epsilon_greedy_policy(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.from_numpy(state).to(self.device)
        return np.array(torch.argmax(self.network(state)).cpu())

    def greedy_policy(self, state):
        state = torch.from_numpy(state).to(self.device)
        return np.array(torch.argmax(self.network(state)).cpu())

    def random_policy(self):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)

    def calculate_TD_targets(self, rewards, next_states, dones):
        target_Q_s = torch.max(self.network(next_states), dim=1)[0] * (~dones)
        td_targets = (rewards + (self.gamma ** self.n_step) * target_Q_s).to(torch.float32)
        return td_targets

    def calculate_Q_values(self, states, actions):
        predicted_Q_values_for_all_actions = self.network(states)
        predicted_Q_values = predicted_Q_values_for_all_actions[torch.arange(actions.size(0)), actions.type(torch.LongTensor)]
        return predicted_Q_values

    def train_network(self, states, actions, rewards, next_states, dones, weights=None):
        self.network.train()
        td_targets = self.calculate_TD_targets(rewards, next_states, dones)
        predicted_Q_values = self.calculate_Q_values(states, actions)

        self.optimizer.zero_grad()
        loss = self.loss_function(predicted_Q_values, td_targets)
        if weights is not None:
            loss = torch.mean(weights*loss)
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        if self.use_PER:
            return torch.abs(predicted_Q_values - td_targets)

    def step(self, state, action, reward, next_state, done):
        self.curr_step += 1
        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.learning_start_buffer_size < self.curr_step and \
                len(self.replay_buffer) > self.batch_size and \
                self.curr_step % self.gradient_update_freq == 0:
            if self.use_PER:
                states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(
                    self.batch_size, device=self.device)
                td_errors = self.train_network(states, actions, rewards, next_states, dones, weights)
                self.replay_buffer.update_TD_errors(indices, td_errors.detach().cpu())
            else:
                self.train_network(*self.replay_buffer.sample(self.batch_size, device=self.device))

