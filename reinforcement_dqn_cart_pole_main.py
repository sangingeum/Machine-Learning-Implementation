from reinforcement.reinforcement_deep_q_network_all_in_one import *
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    total_steps = 50000
    cur_steps = 0
    max_episodes = 10000
    model = reinforcement_deep_q_network_all_in_one(state_size=4, action_size=2, hidden_layer_units=[128, 64, 32],
                                                    use_PER=True, use_double_DQN=True, use_dueling_DQN=True,
                                                    n_step=3, buffer_size=50000)
    for episode in range(max_episodes):
        state = env.reset()[0]
        reward_sum = 0
        while True:
            action = model.epsilon_greedy_policy(state=state)
            next_state, reward, done, _, _ = env.step(action)
            reward_sum += reward
            # reward shaping
            reward -= abs(state[2])*2
            # model step
            model.step(state=state, action=action, reward=reward, next_state=next_state, done=done)
            # update status
            cur_steps += 1
            state = next_state
            if done:
                print("episode: {}, reward sum: {}, steps used: {}, epsilon: {}".format(episode, reward_sum, cur_steps, model.epsilon))
                break
        if cur_steps >= total_steps:
            break
