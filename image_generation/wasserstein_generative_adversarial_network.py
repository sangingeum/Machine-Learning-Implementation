from torch import nn
import numpy as np

class WGANGenerator(nn.Module):
    def __init__(self, img_shape, noise_size, units_per_layer=[128, 256, 512, 1024]):
        super().__init__()
        self.img_shape = img_shape
        self.img_size = np.prod(img_shape)
        self.noise_size = noise_size
        units_per_layer = [noise_size, *units_per_layer, self.img_size]
        self.model = self._build_model(units_per_layer)

    def forward(self, z):
        flattened_img = self.model(z)
        img = flattened_img.view((z.shape[0], *self.img_shape))
        return img

    def _build_model(self, units_per_layer):
        layers = []
        for i in range(len(units_per_layer) - 1):
            layers.append(nn.Linear(units_per_layer[i], units_per_layer[i + 1]))
            layers.append(nn.BatchNorm1d(num_features=units_per_layer[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        layers = layers[:-2]
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)


class WGANDiscriminator(nn.Module):
    def __init__(self, img_shape, units_per_layer=[512, 256, 128]):
        super().__init__()
        self.img_shape = img_shape
        self.img_size = np.prod(img_shape)
        units_per_layer = [self.img_size, *units_per_layer, 1]
        self.model = self._build_model(units_per_layer)

    def forward(self, img):
        img = img.view((img.shape[0], -1))
        return self.model(img)

    def _build_model(self, units_per_layer):
        layers = []
        for i in range(len(units_per_layer) - 1):
            layers.append(nn.Linear(units_per_layer[i], units_per_layer[i+1]))
            layers.append(nn.Dropout(p=0.2))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layers = layers[:-1]
        return nn.Sequential(*layers)