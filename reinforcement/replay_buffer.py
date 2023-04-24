from collections import deque
import torch
import numpy as np

class replay_buffer:
    '''
    transition data buffer
    '''

    def __init__(self, buffer_size=100000, n_step=1, gamma=0.85):
        """
        :param buffer_size: buffer_size, positive integer
        :param n_step: n_step, positive integer
        :param gamma: discount factor, float, used if n>1
        """
        self.buffer_size = buffer_size
        self.n_step = n_step
        self.gamma = gamma

        self.states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.next_states = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)

        self.n_states = deque(maxlen=self.n_step)
        self.n_actions = deque(maxlen=self.n_step)
        self.n_rewards = deque(maxlen=self.n_step)
        self.n_next_states = deque(maxlen=self.n_step)
        self.n_dones = deque(maxlen=self.n_step)

    def __len__(self) -> int:
        return len(self.states)

    def add(self, state, action, reward, next_state, done):
        '''
        add sample to the buffer
        '''
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
                self.states.append(self.n_states[0])
                self.actions.append(self.n_actions[0])
                self.rewards.append(n_step_reward)
                self.next_states.append(next_state)
                self.dones.append(done)
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)

    def sample(self, batch_size, device=None):
        """
        :param batch_size: how many transitions to sample
        :param device: torch.device
        :return:
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sampled_indices = np.random.choice(len(self.states), batch_size, replace=True).astype(np.int)
        sampled_states = torch.tensor(np.array([self.states[i] for i in sampled_indices])).to(device)
        sampled_actions = torch.tensor(np.array([self.actions[i] for i in sampled_indices])).to(device)
        sampled_rewards = torch.tensor(np.array([self.rewards[i] for i in sampled_indices])).to(device)
        sampled_next_states = torch.tensor(np.array([self.next_states[i] for i in sampled_indices])).to(device)
        sampled_dones = torch.tensor(np.array([self.dones[i] for i in sampled_indices])).to(device)

        return sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones
