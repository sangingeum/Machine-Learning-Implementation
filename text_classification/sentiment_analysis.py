import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import deque
class sentiment_analysis:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint).to(self.device).eval()

    def __call__(self, texts: list, batch_size=128, verbose=True, round_result=True):
        """
        :param texts: list of strings
        :param batch_size: batch size
        :param verbose: whether to print progress
        :param round_result: whether to round the result
        :return: class prob or one-hot label depending on 'round_result'
        """

        results = deque()
        # Loop through the texts in batches
        for i in range(0, len(texts), batch_size):
            # Get the batch of texts
            batch_texts = texts[i:i + batch_size]
            # Tokenize the batch of texts
            tokenized_texts = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                # Get the model output for the batch
                model_output = self.model(**tokenized_texts)
            # Get the class probabilities for the batch
            class_prob = torch.nn.functional.softmax(model_output.logits, dim=1)
            # Add the class probabilities to the results list
            results.append(class_prob)
            if verbose:
                print("Processed {} texts".format(i + batch_size))
        # Concatenate the class probabilities for all batches
        class_prob = torch.cat(list(results), dim=0)

        if round_result:
            return torch.round(class_prob)
        else:
            return class_prob
