import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from collections import deque

class recurrent_neural_network_sentiment_analysis(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModel.from_pretrained(self.checkpoint).to(self.device)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity='tanh',
                          batch_first=True, dropout=0, bidirectional=bidirectional).to(self.device)
        if bidirectional:
            self.linear = nn.Linear(in_features=self.hidden_size*2, out_features=1).to(self.device)
        else:
            self.linear = nn.Linear(in_features=self.hidden_size, out_features=1).to(self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor):
        d = 2 if self.bidirectional else 1
        N = X.shape[0]
        h_0 = torch.zeros((self.num_layers * d, N, self.hidden_size)).to(self.device)
        y, h_n = self.rnn(X, h_0)
        y = self.sigmoid(self.linear(y[:, -1, :]))
        return y

    def embed_texts(self, texts: list, batch_size=128):
        results = deque()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokenized_texts = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                model_output = self.model(**tokenized_texts)
                batch_embeddings = model_output["last_hidden_state"]
            results.append(batch_embeddings)

        # return all embeddings
        embeddings = torch.cat(list(results), dim=0)
        return embeddings