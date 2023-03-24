import torch
import multiprocessing
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def train_loop(X: torch.tensor, y: torch.tensor, epochs, test_ratio, model, device, batch_size, loss_function, optimizer, print_interval):
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42, shuffle=True)
    # create data loader
    train_data_set = TensorDataset(X_train, y_train)
    test_data_set = TensorDataset(X_test, y_test)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for i, train_data in enumerate(train_data_loader):
            X, y = train_data
            X = X.to(device)
            y = y.to(device)

            model.train()
            y_prediction = model(X)
            loss = loss_function(y, y_prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if print_interval == 0:
            continue
        if epoch % print_interval == 0:
            model.eval()
            with torch.inference_mode():
                average_test_loss = 0
                for i, test_data in enumerate(test_data_loader):
                    X, y = test_data
                    X = X.to(device)
                    y = y.to(device)
                    test_y_prediction = model(X)
                    test_loss = loss_function(y, test_y_prediction)
                    average_test_loss += test_loss / batch_size
                average_test_loss /= (i+1)
                print("\ntrain loss: {}"
                      "\ntest loss: {}".format(loss / batch_size, average_test_loss))

def get_device_name_agnostic():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_cpu_count():
    return multiprocessing.cpu_count()

