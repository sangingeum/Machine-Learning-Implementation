from reinforcement.reinforcement_deep_q_network_all_in_one import *
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    total_steps = 500000
    cur_steps = 0
    max_episodes = 10000
    model = reinforcement_deep_q_network_all_in_one(state_size=8, action_size=4, use_PER=True, use_double_DQN=True, use_dueling_DQN=True,
                                                    hidden_layer_units=[512, 256, 256], value_layer_units=[128, 64, 32], advantage_layer_units=[128, 64, 32],
                                                    n_step=5, learning_start_buffer_size=100000, buffer_size=200000, epsilon_decay=0.99999, batch_size=512,
                                                    min_epsilon=0.1, target_update_freq=250)
    for episode in range(max_episodes):
        state = env.reset()[0]
        reward_sum = 0
        while True:
            action = model.epsilon_greedy_policy(state=state)
            next_state, reward, done, _, _ = env.step(action)
            reward_sum += reward
            # reward shaping
            reward -= abs(state[0])*2
            # model step
            model.step(state=state, action=action, reward=reward, next_state=next_state, done=done)
            # update status
            cur_steps += 1
            state = next_state
            if done or reward_sum < -250:
                print("episode: {}, reward sum: {}, steps used: {}, epsilon: {}".format(episode, reward_sum, cur_steps, model.epsilon))
                break
        if cur_steps >= total_steps:
            break
