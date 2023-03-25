from supervised.neural_network_multi_label_classification import *
from sklearn.datasets import make_multilabel_classification
from data_preprocessing.normalization import *
from misc.utils import *
from data_preprocessing.class_weighting import *
import numpy as np

def main():
    # device agnostic code
    device = get_device_name_agnostic()
    # load data
    X, y = make_multilabel_classification(n_samples=5000, n_features=30, n_classes=5, n_labels=2,
                                          allow_unlabeled=False)
    # normalize data
    scaler = MaxAbsScaler()
    X = scaler.fit_transform(X)
    # convert to torch.tensor
    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(torch.float32)
    # model, hyper parameters
    model = neural_network_multi_label_classification(units_per_layer=[30, 32, 64, 128, 128, 64, 32, 5]).to(device)
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 1001
    print_interval = 100
    batch_size = 128
    test_ratio = 0.2
    #train loop
    train_loop(X=X, y=y, epochs=epochs, test_ratio=test_ratio, model=model, device=device,
               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer, print_interval=print_interval,
               weighted_sample=False, accuracy_function=round_and_calculate_accuracy)

if __name__ == "__main__":
    main()


