from torch import nn
from reinforcement.replay_buffer import *
from reinforcement.prioritized_replay_buffer_sum_tree import *
from reinforcement.q_network import *
from copy import deepcopy

class reinforcement_deep_q_network_all_in_one:
    """
    All in one DQN
    Current features:
    Double DQN, Dueling DQN, PER, n-step reward
    """
    def __init__(self, state_size, action_size, use_double_DQN=False, use_PER=False, use_dueling_DQN=False,
                 hidden_layer_units=[256, 128, 64], value_layer_units=[128, 64], advantage_layer_units=[128, 64],
                 n_step=1, learning_start_buffer_size=5000, min_epsilon=0.005, epsilon_decay=0.9999, batch_size=128,
                 gamma=0.85, buffer_size=50000, learning_rate=0.0003, gradient_update_freq=1, target_update_freq=16):
        # model creation
        self.state_size = state_size
        self.action_size = action_size
        self.use_dualing_DQN = use_dueling_DQN
        self.network = self._build_model(use_dueling_DQN, hidden_layer_units, value_layer_units, advantage_layer_units)
        # hyper parameter setting
        self.learning_start_buffer_size = learning_start_buffer_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.curr_step = 0

        self.epsilon = 1.0
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.learning_rate = learning_rate
        self.gradient_update_freq = gradient_update_freq
        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.HuberLoss()

        self.use_double_DQN = use_double_DQN
        if use_double_DQN:
            self.target_update_freq = target_update_freq
            self.calculate_TD_targets = self.calculate_TD_targets_double
            self.calculate_Q_values = self.calculate_Q_values_double
        else:
            self.calculate_TD_targets = self.calculate_TD_targets_basic
            self.calculate_Q_values = self.calculate_Q_values_basic

        self.use_PER = use_PER
        if use_PER:
            self.replay_buffer = prioritized_replay_buffer(buffer_size=self.buffer_size, n_step=self.n_step, gamma=self.gamma)
        else:
            self.replay_buffer = replay_buffer(buffer_size=self.buffer_size, n_step=self.n_step, gamma=self.gamma)

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_network = deepcopy(self.network).to(self.device)
        self.network = self.network.to(self.device)

    def _build_model(self, use_dualing_DQN, hidden_layer_units, value_layer_units, advantage_layer_units):
        if use_dualing_DQN:
            return dueling_q_network(self.state_size, self.action_size, hidden_layer_units, value_layer_units, advantage_layer_units)
        else:
            return basic_q_network(self.state_size, self.action_size, hidden_layer_units)

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

    def calculate_TD_targets_double(self, actions, rewards, next_states, dones):
        target_Q_s = self.target_network(next_states)
        target_actions = torch.argmax(self.network(next_states), dim=1)
        max_target_Q_s = target_Q_s[torch.arange(actions.size(0)), target_actions]
        max_target_Q_s = max_target_Q_s * (~dones)
        td_targets = (rewards + (self.gamma ** self.n_step) * max_target_Q_s).to(torch.float32)
        return td_targets

    def calculate_Q_values_double(self, states, actions):
        predicted_Q_values_for_all_actions = self.network(states)
        predicted_Q_values = predicted_Q_values_for_all_actions[torch.arange(actions.size(0)), actions.type(torch.LongTensor)]
        return predicted_Q_values

    def calculate_TD_targets_basic(self, actions, rewards, next_states, dones):
        target_Q_s = torch.max(self.network(next_states), dim=1)[0] * (~dones)
        td_targets = (rewards + (self.gamma ** self.n_step) * target_Q_s).to(torch.float32)
        return td_targets

    def calculate_Q_values_basic(self, states, actions):
        predicted_Q_values_for_all_actions = self.network(states)
        predicted_Q_values = predicted_Q_values_for_all_actions[torch.arange(actions.size(0)), actions.type(torch.LongTensor)]
        return predicted_Q_values

    def train_network(self, states, actions, rewards, next_states, dones, weights=None):
        self.network.train()
        self.target_network.train()
        td_targets = self.calculate_TD_targets(actions, rewards, next_states, dones)
        predicted_Q_values = self.calculate_Q_values(states, actions)
        self.optimizer.zero_grad()
        loss = self.loss_function(predicted_Q_values, td_targets)
        if weights is not None:
            loss = torch.mean(weights*loss)
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.update_target_network()
        if self.use_PER:
            return torch.abs(predicted_Q_values - td_targets)

    def update_target_network(self):
        if self.use_double_DQN and (self.curr_step % self.target_update_freq) == 0:
            self.target_network = deepcopy(self.network)

    def step(self, state, action, reward, next_state, done):
        self.curr_step += 1
        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.learning_start_buffer_size < self.curr_step and \
                len(self.replay_buffer) > self.batch_size and \
                self.curr_step % self.gradient_update_freq == 0:
            if self.use_PER:
                states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size, device=self.device)
                td_errors = self.train_network(states, actions, rewards, next_states, dones, weights)
                self.replay_buffer.update_priorities(indices, td_errors.detach().cpu())
            else:
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, device=self.device)
                self.train_network(states, actions, rewards, next_states, dones)

