from torchtext.datasets import IMDB
from torch.utils.data import Dataset
import random
from text_classification.recurrent_neural_network_sentiment_analysis import *
from misc.utils import *

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

"""
def collate(samples):
    # Get the batch size
    batch_size = len(samples)

    # Determine the maximum sequence length and input feature size
    max_seq_len = max([sample[0].shape[1] for sample in samples])
    feat_size = samples[0][0].shape[2]

    # Initialize the output tensors
    inputs = torch.zeros((batch_size, max_seq_len, feat_size))
    labels = torch.FloatTensor([sample[1] for sample in samples]).reshape((-1, 1))

    # Pad the input tensors and copy them to the output tensor
    for i, sample in enumerate(samples):
        seq_len = sample[0].shape[1]
        inputs[i, :seq_len, :] = sample[0]

    return inputs, labels
"""

if __name__ == "__main__":
    device = get_device_name_agnostic()
    model = recurrent_neural_network_sentiment_analysis(768, 512, bidirectional=False).to(device)

    train_iter = IMDB(root='./data', split='train')
    test_iter = IMDB(root='./data', split='test')

    y_train, X_train = zip(*random.sample(list(train_iter), k=5000))
    y_test, X_test = zip(*random.sample(list(test_iter), k=5000))

    # convert label to tensor
    y_train = torch.reshape(torch.tensor(y_train, dtype=torch.float32) - 1, (-1, 1))
    y_test = torch.reshape(torch.tensor(y_test, dtype=torch.float32) - 1, (-1, 1))
    # convert text to vector
    train_data_set = TextDataset(X_train, y_train)
    test_data_set = TextDataset(X_test, y_test)

    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 301
    print_interval = 1
    batch_size = 512

    # train loop
    train_loop(train_data_set=train_data_set, test_data_set=test_data_set, epochs=epochs, model=model, device=device,
               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer, print_interval=print_interval,
               weighted_sample=False, accuracy_function=calculate_accuracy_binary_class, X_on_the_fly_function=model.embed_texts,
               test_first=True)


