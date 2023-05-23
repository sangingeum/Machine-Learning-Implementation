import numpy as np
from collections import defaultdict
from collections import deque

class reinforcement_monte_carlo():

    def __init__(self, env, num_iterations=10000,
                 epsilon=1.0, min_epsilon=0.01, epsilon_decay_rate=0.9995,
                 alpha=0.001, gamma=0.99):
        self.env = env
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.alpha = alpha
        self.gamma = gamma
        self.action_size = self.env.action_space.n
        # optimistic initialization
        self.Q = defaultdict(lambda: 100*np.ones(self.action_size))

    def fit(self, num_iterations=None):
        if num_iterations is None:
            num_iterations = self.num_iterations
        episode_buffer = deque()
        for i in range(num_iterations):
            state = self.env.reset()[0]
            episode_buffer.clear()
            while True:
                action = self.epsilon_greedy_policy(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_buffer.append((state, action, reward))
                state = next_state
                if done:
                    if i % 1000 == 0:
                        print("training itration {}, epsilon {}".format(i, self.epsilon))
                    Return = 0
                    while len(episode_buffer) > 0:
                        s, a, r = episode_buffer.pop()
                        Return = self.gamma * Return + r
                        self.Q[s][a] += self.alpha * (Return - self.Q[s][a])
                    self.update_epsilon()
                    break

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.min_epsilon)


    def epsilon_greedy_policy(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.Q[state])

    def greedy_policy(self, state):
        return np.argmax(self.Q[state])

