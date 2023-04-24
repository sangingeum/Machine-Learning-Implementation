from torch import nn
from reinforcement.replay_buffer import *
from copy import deepcopy

class reinforcement_double_deep_q_network:
    """
    n-step DDQN with a replay buffer
    """
    def __init__(self, units_per_layer=[4, 64, 128, 64, 2]):
        # model creation
        layers = list()
        layer_len = len(units_per_layer)
        if layer_len < 2:
            raise AssertionError
        for i in range(layer_len - 1):
            layers.append(nn.Linear(units_per_layer[i], units_per_layer[i + 1]))
            if i < layer_len - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

        # hyper parameter setting
        self.state_size = units_per_layer[0]
        self.action_size = units_per_layer[-1]
        self.curr_step = 0
        self.learning_rate = 0.0003
        self.buffer_size = 50000
        self.learning_start_buffer_size = 5000
        self.batch_size = 128
        self.epsilon = 1.0
        self.min_epsilon = 0.005
        self.epsilon_decay = 0.9999
        self.gamma = 0.85
        self.n_step = 3
        self.target_update_freq = 16
        self.gradient_update_freq = 1
        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.HuberLoss()
        self.replay_buffer = replay_buffer(buffer_size=self.buffer_size, n_step=self.n_step, gamma=self.gamma)

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_network = deepcopy(self.network).to(self.device)
        self.network = self.network.to(self.device)

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

    def train_network(self, states, actions, rewards, next_states, dones):
        self.network.train()
        self.target_network.train()

        target_Q_s = self.target_network(next_states)
        target_actions = torch.argmax(self.network(next_states), dim=1)

        max_target_Q_s = target_Q_s[torch.arange(actions.size(0)), target_actions]
        max_target_Q_s = max_target_Q_s * (~dones)
        td_targets = (rewards + (self.gamma ** self.n_step) * max_target_Q_s).to(torch.float32)

        Q_s = self.network(states)
        Q_sa = Q_s[torch.arange(actions.size(0)), actions.type(torch.LongTensor)]

        self.optimizer.zero_grad()
        loss = self.loss_function(Q_sa, td_targets)
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        if self.curr_step % self.target_update_freq == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_network = deepcopy(self.network)

    def step(self, state, action, reward, next_state, done):
        self.curr_step += 1
        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.learning_start_buffer_size < self.curr_step and \
                len(self.replay_buffer) > self.batch_size and \
                self.curr_step % self.gradient_update_freq == 0:
            self.train_network(*self.replay_buffer.sample(self.batch_size, device=self.device))

