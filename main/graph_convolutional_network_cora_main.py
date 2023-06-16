import torch
from misc.utils import *
from graph_classification.graph_convolutional_network_node_classification import *
from torch.utils.data import Dataset
from graph_classification.cora_dataset import *

class IndexDataset(Dataset):
    def __init__(self, indexes):
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.indexes[idx]

if __name__ == "__main__":
    # Load cora data
    features, labels, adjacency_matrix = load_cora_data(add_eye=True, sparse=True, normalize_adj=True, one_hot=True)
    # Create model
    device = get_device_name_agnostic()
    model = graph_convolutional_network_node_classification(input_size=features.shape[1], output_size=7, drop_out=0.5).to(device)
    # Set hyper parameters
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 201
    print_interval = 10
    batch_size = 128
    # split data
    indexes = torch.randperm(len(features))
    sep = int(0.20*len(features)) # use only 20% for training
    train_index = indexes[:sep]
    test_index = indexes[sep:]
    # make data set
    train_data_set = IndexDataset(train_index)
    test_data_set = IndexDataset(test_index)
    # train loop
    train_loop_with_adjacency_matrix(train_data_set=train_data_set, test_data_set=test_data_set, epochs=epochs,
                                     features=features, labels=labels,
                                     model=model, device=device, batch_size=batch_size, loss_function=loss_function,
                                     optimizer=optimizer, print_interval=print_interval, adjacency_matrix=adjacency_matrix)
