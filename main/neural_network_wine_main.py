from tabular_classification.neural_network_multi_class_classification import *
from sklearn.datasets import load_wine
from data_preprocessing.normalization import *
from misc.utils import *

def main():
    # device agnostic code
    device = get_device_name_agnostic()
    # load data
    X, y = load_wine(return_X_y=True, as_frame=False)
    # normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # convert to torch
    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(torch.long)
    # model, hyper parameters
    model = neural_network_multi_class_classification(units_per_layer=[13, 32, 64, 32, 3]).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 301
    print_interval = 15
    batch_size = 128
    test_ratio = 0.2

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=True)

    # make data set
    train_data_set = TensorDataset(X_train, y_train)
    test_data_set = TensorDataset(X_test, y_test)

    #train loop
    train_loop(train_data_set=train_data_set, test_data_set=test_data_set, epochs=epochs, model=model, device=device,
               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer, print_interval=print_interval,
               weighted_sample=False, calculate_accuracy=True)

if __name__ == "__main__":
    main()


