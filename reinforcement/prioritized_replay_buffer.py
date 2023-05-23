from collections import deque
import torch
from reinforcement.sum_tree import *

class prioritized_replay_buffer:
    '''
    prioritized replay buffer based on the sum tree structure
    '''

    def __init__(self, buffer_size=100000, n_step=1, gamma=0.85,
                 alpha=0.8, beta=0.3, beta_increment=0.0005
                 ):
        """
        :param buffer_size: buffer_size, positive integer
        :param n_step: n_step, positive integer
        :param gamma: discount factor, float, used if n_step>1
        """
        self.buffer_size = buffer_size
        self.n_step = n_step
        self.gamma = gamma
        self.sum_tree = sum_tree(tree_size=buffer_size)

        if self.n_step > 1:
            self.n_states = deque(maxlen=self.n_step)
            self.n_actions = deque(maxlen=self.n_step)
            self.n_rewards = deque(maxlen=self.n_step)
            self.n_next_states = deque(maxlen=self.n_step)
            self.n_dones = deque(maxlen=self.n_step)

        self.default_TD_error = 1
        self.small_positive_value = 1e-5
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

    def __len__(self) -> int:
        return self.buffer_size

    def add(self, state, action, reward, next_state, done):
        '''
        add sample to the buffer
        '''

        initial_priority = self.sum_tree.get_initial_priority()
        # n-step
        if self.n_step > 1:
            self.n_states.append(state)
            self.n_actions.append(action)
            self.n_rewards.append(reward)
            self.n_next_states.append(next_state)
            self.n_dones.append(done)

            if len(self.n_states) == self.n_step:
                n_step_reward = 0
                for i in range(self.n_step):
                    n_step_reward += (self.gamma ** i) * self.n_rewards[i]

                data = (self.n_states[0], self.n_actions[0], n_step_reward, next_state, done)
                self.sum_tree.add(initial_priority, data)

            if done:
                self.n_states.clear()
                self.n_actions.clear()
                self.n_rewards.clear()
                self.n_next_states.clear()
                self.n_dones.clear()
        else:
            data = (state, action, reward, next_state, done)
            self.sum_tree.add(initial_priority, data)

    def update_priorities(self, indices, new_priorities):
        new_priorities = np.power(new_priorities, self.alpha)
        for i in range(len(indices)):
            self.sum_tree.update(indices[i], new_priorities[i])

    def sample(self, batch_size, device=None):
        """
        :param batch_size: how many transitions to sample
        :param device: torch.device
        :return:
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # sample data
        samples = deque()
        sum_size = float(self.sum_tree.get_total_sum() - self.small_positive_value)
        segment_size = sum_size / batch_size

        for i in range(batch_size):
            start = max(i * segment_size, self.small_positive_value)
            end = min((i+1) * segment_size, sum_size)
            target_sum = np.random.uniform(low=start, high=end)
            sample = self.sum_tree.get(target_sum)
            samples.append(sample)

        sampled_indices, sampled_priorities, sampled_transitions = zip(*list(samples))
        sampled_indices = np.array(sampled_indices)
        sampled_priorities = np.array(sampled_priorities)

        # calculate sampled_weights
        sampled_prob = (sampled_priorities + self.small_positive_value) / sum_size
        sampled_weights = torch.from_numpy(np.power(self.sum_tree.tree_size * sampled_prob, -self.beta)).to(device)

        # convert to tensor
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = zip(*sampled_transitions)
        sampled_states = torch.tensor(np.array(sampled_states)).to(device)
        sampled_actions = torch.tensor(np.array(sampled_actions)).to(device)
        sampled_rewards = torch.tensor(np.array(sampled_rewards)).to(device)
        sampled_next_states = torch.tensor(np.array(sampled_next_states)).to(device)
        sampled_dones = torch.tensor(np.array(sampled_dones)).to(device)

        # update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones, sampled_indices, sampled_weights
