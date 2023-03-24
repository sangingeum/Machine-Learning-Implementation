from sklearn.preprocessing import LabelBinarizer


"""
Example:
    # assume label is an integer between 0 and 2
    label = 1
    
    # determine number of classes
    num_classes = 3
    
    # create one-hot label
    lb = LabelBinarizer()
    one_hot_label = lb.fit_transform([label])
    
    print(one_hot_label)


"""