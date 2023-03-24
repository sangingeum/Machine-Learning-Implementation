from supervised.neural_network_regression import *
from sklearn.datasets import load_diabetes
from data_preprocessing.normalization import *
from misc.utils import *

def main():
    # device agnostic code
    device = get_device_name_agnostic()
    # load data
    X, y = load_diabetes(return_X_y=True, as_frame=False)
    y = y.reshape((-1, 1))
    # normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # convert to torch.tensor
    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(torch.float32)
    # model, hyper parameters
    model = neural_network_regression(units_per_layer=[10, 16, 32, 16, 1]).to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 2001
    print_interval = 100
    batch_size = 256
    test_ratio = 0.2
    #train loop
    train_loop(X=X, y=y, epochs=epochs, test_ratio=test_ratio, model=model, device=device,
               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer, print_interval=print_interval)


if __name__ == "__main__":
    main()


