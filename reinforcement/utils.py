import numpy as np

def create_FrozenLake_v1_4x4_slippery_MDP():
    state_size = 16
    action_size = 4
    action_adder_table = [[0, -1], [1, 0], [0, 1], [-1, 0]]
    actual_action_table = [[0, 1, 3], [0, 1, 2], [1, 2, 3], [0, 2, 3]]
    terminal_states = [5, 7, 11, 12, 15]
    slippery_MDP = [[[] for j in range(4)] for i in range(16)]

    def next_state(state, actual_a):
        row = state // 4
        col = state % 4
        next_row = row + action_adder_table[actual_a][0]
        next_col = col + action_adder_table[actual_a][1]
        if (next_row >= 0 and next_col >= 0 and next_row <= 3 and next_col <= 3):
            return int(next_row * 4 + next_col)
        else:
            return int(row * 4 + col)

    for s in range(state_size):
        for a in range(action_size):
            if s in terminal_states:
                slippery_MDP[s][a].append([1.0, s, 0.0])
            else:
                for i, actual_a in enumerate(actual_action_table[a]):
                    next_s = next_state(s, actual_a)
                    slippery_MDP[s][a].append([1.0 / 3.0, next_s, 1.0 if next_s == state_size - 1 else 0.0])
    return slippery_MDP