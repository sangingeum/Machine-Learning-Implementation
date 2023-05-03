from sklearn.preprocessing import LabelBinarizer
import torch.nn.functional as F

"""
Example1:
    # assume label is an integer between 0 and 2
    label = 1
    # determine number of classes
    num_classes = 3
    # create one-hot label
    lb = LabelBinarizer()
    one_hot_label = lb.fit_transform([label])
    print(one_hot_label)
Example2:
    A = torch.tensor([3, 4, 5, 0, 1, 2])
    output = F.one_hot(A, num_classes = 7)
    print(output)
"""