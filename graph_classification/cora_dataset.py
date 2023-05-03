import torch
import numpy as np
from datasets import load_dataset
from collections import deque
import torch.nn.functional as F
from collections import defaultdict


def load_cora_data(add_eye=True, sparse=False, normalize_adj=True, one_hot=True):
    datasets = load_dataset('gcaillaut/cora', name="nodes", split="train")
    column_names = datasets.column_names
    datasets = datasets[:]
    q = deque()
    for column_name in column_names[:-3]:
        col = datasets[column_name]
        q.append(np.array(col, dtype=int).reshape(-1, 1))
    # extract columns
    feature_columns = torch.from_numpy(np.hstack(list(q))).type(torch.FloatTensor)
    node_column = datasets[column_names[-3]]
    label_column = torch.LongTensor(np.array(datasets[column_names[-2]]))
    neighbors_column = datasets[column_names[-1]]
    if one_hot:
        label_column = F.one_hot(label_column).type(torch.float32)
    # create node_to_index dict
    node_to_index = defaultdict()
    for i, node in enumerate(node_column):
        node_to_index[node] = i

    # create adjacency matrix
    n = len(node_column)
    if sparse:
        indices = deque()
        for i, target_nodes in enumerate(neighbors_column):
            if add_eye:
                indices.append([i, i])
            for target_node in target_nodes:
                target_index = node_to_index[target_node]
                # undirected edge
                indices.append([i, target_index])
                indices.append([target_index, i])
        data = torch.ones(len(indices))
        indices = torch.LongTensor(list(indices))
        adjacency_matrix = torch.sparse_coo_tensor(indices.t(), data, size=(n, n))
    else:
        if add_eye:
            adjacency_matrix = torch.eye(n)
        else:
            adjacency_matrix = torch.zeros((n, n))
        for i, target_nodes in enumerate(neighbors_column):
            for target_node in target_nodes:
                target_index = node_to_index[target_node]
                # undirected edge
                adjacency_matrix[i][target_index] = 1
                adjacency_matrix[target_index][i] = 1

    # normalize adjacency_matrix
    if normalize_adj:
        degrees = torch.pow(torch.sum(adjacency_matrix, dim=0).to_dense(), -0.5)
        degree_matrix = torch.zeros((n, n))
        for i, degree in enumerate(degrees):
            degree_matrix[i][i] = degree
        adjacency_matrix = torch.sparse.mm(torch.sparse.mm(degree_matrix, adjacency_matrix), degree_matrix)

    return feature_columns, label_column, adjacency_matrix