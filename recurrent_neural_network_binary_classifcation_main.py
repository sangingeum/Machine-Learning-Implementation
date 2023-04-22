from torchtext.datasets import IMDB
import random
from text_classification.recurrent_neural_network_sentiment_analysis import *
from misc.utils import *

if __name__ == "__main__":
    device = get_device_name_agnostic()
    model = recurrent_neural_network_sentiment_analysis(768, 512, bidirectional=False).to(device)
    #
    train_iter = IMDB(root='./data', split='train')
    test_iter = IMDB(root='./data', split='test')

    y_train, X_train = zip(*random.sample(list(train_iter), k=1000))
    y_test, X_test = zip(*random.sample(list(test_iter), k=1000))
    # convert label to tensor
    y_train = torch.reshape(torch.tensor(y_train, dtype=torch.float32) - 1, (-1, 1)).to(device)
    y_test = torch.reshape(torch.tensor(y_test, dtype=torch.float32) - 1, (-1, 1)).to(device)
    # convert text to vector
    X_train = model.embed_texts(X_train, batch_size=128).to(device)
    X_test = model.embed_texts(X_test, batch_size=128).to(device)
    # make data set
    train_data_set = TensorDataset(X_train, y_train)
    test_data_set = TensorDataset(X_test, y_test)

    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 301
    print_interval = 1
    batch_size = 128

    # train loop
    train_loop(train_data_set=train_data_set, test_data_set=test_data_set, epochs=epochs, model=model, device=device,
               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer, print_interval=print_interval,
               weighted_sample=False, accuracy_function=calculate_accuracy_binary_class)


