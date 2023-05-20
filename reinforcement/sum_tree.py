import numpy as np
class sum_tree:

    def __init__(self, tree_size):
        self.tree_size = tree_size
        self.tree = np.zeros((2 * self.tree_size - 1,), dtype=np.float64)
        self.data = np.zeros((self.tree_size,), dtype=object)
        self.write_index = 0
        self.num_entries = 0
        self.highest_priority = 1.0
    def _propagate(self, index, change):
        # propagate change starting from index to the root
        parent = (index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _search(self, index, sum):
        left = 2 * index + 1
        right = left + 1

        if left >= len(self.tree):
            return index

        if sum <= self.tree[left]:
            return self._search(left, sum)
        else:
            return self._search(right, sum - self.tree[left])

    def add(self, priority, data):
        # get tree index
        index = self.write_index + self.tree_size - 1
        # update priority and data
        self.update(index, priority)
        self.data[self.write_index] = data
        # update counters
        self.write_index += 1
        if self.write_index >= self.tree_size:
            self.write_index = 0
        if self.num_entries < self.tree_size:
            self.num_entries += 1
        # update the highest priority
        self.highest_priority = max(self.highest_priority, priority)

    def update(self, index, priority):
        # assign priority & propagate change
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, change)

    def get(self, sum):
        # retrieve data at sum
        target_index = self._search(0, sum)
        dataIdx = min(target_index - self.tree_size + 1, self.num_entries-1)
        return (target_index, self.tree[target_index], self.data[dataIdx])

    def get_total_sum(self):
        return self.tree[0]

    def get_initial_priority(self):
        return self.highest_priority

