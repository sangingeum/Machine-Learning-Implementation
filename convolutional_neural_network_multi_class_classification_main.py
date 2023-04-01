from supervised.convolutional_neural_network_multi_class_classification import *
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import ToTensor
from data_preprocessing.normalization import *
from misc.utils import *

def main():
    # device agnostic code
    device = get_device_name_agnostic()
    # transforamtions
    transforamtions = transforms.Compose([
        ToTensor(),
        Normalize(0.5, 0.5)
    ])
    # load data
    train_data_set = CIFAR10(root="data", train=True, download=True, transform=transforamtions)
    test_data_set = CIFAR10(root="data", train=False, download=True, transform=transforamtions)

    # model, hyper parameters
    model = convolutional_neural_network_multi_class_classification(32, 32, channels=[3, 32, 128],
                                                                    units_in_hidden_layers=[128, 64, 32, 10]).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 501
    print_interval = 1
    batch_size = 1024

    #train loop
    train_loop(train_data_set=train_data_set, test_data_set=test_data_set, epochs=epochs, model=model, device=device,
               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer, print_interval=print_interval,
               weighted_sample=False, accuracy_function=calculate_accuracy_multi_class)

if __name__ == "__main__":
    main()


