import torch
from torch import nn
from misc.utils import *

class convolutional_neural_network_multi_class_classification(nn.Module):
    def __init__(self, height, width, channels=[1, 16, 32, 64], units_in_hidden_layers=[128, 64, 32, 10]):
        super().__init__()

        if height <= 0 or width <= 0:
            raise Exception("height and width should be positive.")
        if len(units_in_hidden_layers) < 2:
            raise Exception("units_in_hidden_layers should be longer than 2.")
        if len(channels) < 2:
            raise Exception("channels should be longer than 2.")

        conv_2d_kernel_size = 3
        conv_2d_stride = 1
        conv_2d_padding = 1
        max_pool_2d_kernel_size = 2
        max_pool_2d_stride = 2

        layers = []
        # Convolutional layers
        for i in range(len(channels)-1):
            layers.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1],
                                    kernel_size=conv_2d_kernel_size, padding=conv_2d_padding,
                                    stride=conv_2d_stride, bias=True))
            height, width = calculate_height_width_after_conv2d(height, width, kernel_size=conv_2d_kernel_size,
                                                                stride=conv_2d_stride, padding=conv_2d_padding)
            layers.append(nn.Dropout(p=0.2))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=max_pool_2d_kernel_size, stride=max_pool_2d_stride))
            height, width = calculate_height_width_after_max_pool_2d(height, width, kernel_size=max_pool_2d_kernel_size,
                                                                     stride=max_pool_2d_stride)

        # Flatten
        layers.append(nn.Flatten())

        units_in_hidden_layers = [width*height*channels[-1]] + units_in_hidden_layers
        # ANN
        for i in range(len(units_in_hidden_layers)-1):
            layers.append(nn.Linear(units_in_hidden_layers[i], units_in_hidden_layers[i + 1]))
            if i < len(units_in_hidden_layers) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=0.2))

        layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)





