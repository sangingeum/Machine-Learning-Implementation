import gymnasium as gym
from reinforcement.reinforcement_monte_carlo import *

def main():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    model = reinforcement_monte_carlo(env=env, num_iterations=10000,
                                      epsilon_decay_rate=0.9999, alpha=0.01, gamma=0.9)
    model.fit()
    episodes = 100
    return_sum = 0
    for episode in range(episodes):
        state = env.reset()[0]
        ret = 0
        while True:
            action = model.epsilon_greedy_policy(state)
            next_state, reward, done, _, _ = env.step(action)
            ret += reward
            state = next_state
            if done:
                return_sum += ret
                print("Episode {}, Return {}".format(episode, ret))
                break
    print()
    print("Average return: {}".format(return_sum/episodes))


if __name__ == "__main__":
    main()
