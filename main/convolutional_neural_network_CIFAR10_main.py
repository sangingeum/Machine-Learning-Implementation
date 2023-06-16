from image_classification.convolutional_neural_network_multi_class_classification import *
from torchvision.datasets import CIFAR10
from torchvision import transforms
from misc.utils import *

def main():
    # device agnostic code
    device = get_device_name_agnostic()
    # transformations
    train_transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    test_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    # load data
    train_data_set = CIFAR10(root="data", train=True, download=True, transform=train_transformations)
    test_data_set = CIFAR10(root="data", train=False, download=True, transform=test_transformations)

    # model, hyper parameters
    model = convolutional_neural_network_multi_class_classification(32, 32, channels=[3, 32, 64, 128],
                                                                    units_in_hidden_layers=[128, 64, 32, 10]).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    epochs = 501
    print_interval = 1
    batch_size = 1024

    # train loop
    train_loop(train_data_set=train_data_set, test_data_set=test_data_set, epochs=epochs, model=model, device=device,
               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer, print_interval=print_interval,
               weighted_sample=False, calculate_accuracy=True)

if __name__ == "__main__":
    main()


