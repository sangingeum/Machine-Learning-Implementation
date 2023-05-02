from tabular_classification.bagging_and_voting_classification import *
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # load data
    X, y = load_wine(return_X_y=True, as_frame=False)
    # create model
    model = bagging_and_voting_classification(n_estimators=20, max_samples=0.5, max_features=0.5,
                 use_MLPClassifier=True, use_DecisionTreeClassifier=True, use_SVC=False, soft_voting=True,
                 MLP_hidden_layer_sizes=[15, 10], MLP_max_iter=3500,
                 DecisionTree_max_depth=5, SVC_kernel='linear', SVC_C=0.1)
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    # fit the model
    model.fit(X_train=X_train, y_train=y_train)
    # predict
    y_pred = model(X_test)
    # calculate accuracy
    print(y_pred)
    print(y_test)
    print("Accuracy: {}".format((y_pred==y_test).sum()/len(y_pred)))
