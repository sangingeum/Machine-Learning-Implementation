import numpy as np

class reinforcement_value_iteration():
    def __init__(self, state_size, action_size, MDP, theta=0.01):
        """
        :param state_size: integer
        :param action_size: integer
        :param MDP: MDP[state][action] = [[prob1, next_state1, reward1],
                                          [prob2, next_state2, reward2],
                                          [prob3, next_state3, reward3],
                                           ...]
        :param theta: A small value used to stop value iteration
        """
        self.state_size = state_size
        self.action_size = action_size
        self.MDP = MDP
        self.theta = theta
        self.value = np.zeros(self.state_size)
        self.policy = np.zeros(self.state_size, dtype="int32")

    def fit(self):
        while True:
            delta = 0
            for s in range(self.state_size):
                prev_value = self.value[s]
                max_value = 0
                for a in range(self.action_size):
                    cur_value = 0
                    for (prob, next_state, reward) in self.MDP[s][a]:
                        cur_value += prob * (reward + self.value[next_state])
                    if cur_value > max_value:
                        max_value = cur_value
                        self.policy[s] = a
                self.value[s] = max_value
                delta = max(delta, abs(prev_value - max_value))
            if delta < self.theta:
                break

    def greedy_policy(self, state):
        return self.policy[state]