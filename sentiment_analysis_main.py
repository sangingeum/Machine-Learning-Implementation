from text_classification.sentiment_analysis import *
from torchtext.datasets import IMDB
import torch

if __name__ == "__main__":
    # load IMDB data
    train_iter = IMDB(root='./data', split='test')
    # separate label and text
    labels, texts = zip(*list(train_iter))
    #labels = labels[:1000]
    #texts = texts[:1000]
    # make label 0 or 1
    labels = torch.tensor(labels) - 1
    analyzer = sentiment_analysis()
    # predict the sentiment of given texts.
    # If GPU mem is not enough, adjust 'batch_size'.
    predictions = torch.argmax(analyzer(list(texts), batch_size=128, round_result=True, verbose=True), dim=1).cpu()
    accuracy = torch.sum(labels == predictions) / len(labels)
    # Accuracy: 0.890720009803772
    print("Accuracy: {}".format(accuracy))