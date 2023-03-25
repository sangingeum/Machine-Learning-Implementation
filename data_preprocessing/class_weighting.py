import torch
from torch.utils.data import WeightedRandomSampler


def get_weighted_sampler(labels):
    # Check input tensor shape
    if len(labels.shape) != 2:
        raise ValueError("Input tensor must be 2D")
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


