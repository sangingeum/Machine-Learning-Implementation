import torch

from supervised.convolutional_neural_network_multi_class_classification import *
from supervised.neural_network_multi_class_classification import *
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from data_preprocessing.normalization import *
from data_preprocessing.one_hot_encoding import *
from misc.utils import *


def main():
    # device agnostic code
    device = get_device_name_agnostic()
    # load data
    train_data = FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
    # concatenate data
    X = torch.cat([train_data.data, test_data.data], dim=0).type(torch.float32).unsqueeze(dim=1)
    y = torch.cat([train_data.targets, test_data.targets], dim=0).type(torch.float32)
    # normalize data
    transform = Normalize(0, 1)
    X = transform(X)
    # convert to one hot labels
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    # convert to torch.tensor
    y = torch.from_numpy(y).type(torch.float32)
    # model, hyper parameters
    model = convolutional_neural_network_multi_class_classification(28, 28, channels=[1, 32, 128],
                                                                    units_in_hidden_layers=[128, 64, 32, 10]).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 501
    print_interval = 50
    batch_size = 1024
    test_ratio = 0.2
    #train loop
    train_loop(X=X, y=y, epochs=epochs, test_ratio=test_ratio, model=model, device=device,
               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer, print_interval=print_interval,
               weighted_sample=False, accuracy_function=round_and_calculate_accuracy)

if __name__ == "__main__":
    main()


