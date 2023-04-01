import torch
from torch.utils.data import WeightedRandomSampler


def get_weighted_sampler(labels):

    # Check input tensor shape
    if len(labels.shape) == 1:
        labels = torch.reshape(labels, (-1, 1))
    if len(labels.shape) >= 3:
        raise ValueError("Input tensor must be lower than 3D, {}-D given".format(len(labels.shape)))

    # Compute class frequencies
    uniques, counts = torch.unique(labels, dim=0, return_counts=True)
    class_weights = labels.shape[0] / counts
    # Compute sample weights
    sample_weights = torch.zeros(labels.shape[0], dtype=torch.float32)
    for i, unique in enumerate(uniques):
        sample_weights[torch.all(labels == unique, dim=1)] = class_weights[i]
    # Create weighted sampler
    sampler = WeightedRandomSampler(sample_weights, labels.shape[0])

    return sampler


"""
    Example:
    train_sampler = get_weighted_sampler(y_train)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, sampler=train_sampler)


"""

