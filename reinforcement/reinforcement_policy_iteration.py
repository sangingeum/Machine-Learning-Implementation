import numpy as np

class reinforcement_policy_iteration():
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

    def policy_evaluation(self):
        while True:
            delta = 0
            for s in range(self.state_size):
                prev_value = self.value[s]
                eval_value = 0
                a = self.policy[s]
                for (prob, next_state, reward) in self.MDP[s][a]:
                    eval_value += prob * (reward + self.value[next_state])
                self.value[s] = eval_value
                delta = max(delta, abs(prev_value - eval_value))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for s in range(self.state_size):
            prev_action = self.policy[s]
            max_value = 0
            new_action = 0
            for a in range(self.action_size):
                cur_value = 0
                for (prob, next_state, reward) in self.MDP[s][a]:
                    cur_value += prob * (reward + self.value[next_state])
                if cur_value > max_value:
                    max_value = cur_value
                    new_action = a
            if prev_action != new_action:
                self.policy[s] = new_action
                policy_stable = False
        return policy_stable

    def fit(self):
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            if policy_stable:
                break

    def greedy_policy(self, state):
        return self.policy[state]