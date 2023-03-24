from supervised.neural_network_multi_class_classification import *
from sklearn.datasets import load_wine
from data_preprocessing.normalization import *
from data_preprocessing.one_hot_encoding import *
from misc.utils import *


def main():
    # device agnostic code
    device = get_device_name_agnostic()
    # load data
    X, y = load_wine(return_X_y=True, as_frame=False)
    y = y.reshape((-1, 1))
    # normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # convert to one hot labels
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    # convert to torch.tensor
    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(torch.float32)

    # model, hyper parameters
    model = neural_network_multi_class_classification(units_per_layer=[13, 32, 64, 32, 3]).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 301
    print_interval = 15
    batch_size = 128
    test_ratio = 0.2
    #train loop
    train_loop(X=X, y=y, epochs=epochs, test_ratio=test_ratio, model=model, device=device,
               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer, print_interval=print_interval,
               accuracy_function=calculate_multi_class_accuracy)

if __name__ == "__main__":
    main()


