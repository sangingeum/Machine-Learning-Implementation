import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from scipy.stats import mode
class bagging_and_voting_classification:
    def __init__(self, n_estimators=10, max_samples=0.5, max_features=0.5,
                 use_MLPClassifier=True, use_DecisionTreeClassifier=False, use_SVC=False,
                 soft_voting=True,
                 MLP_hidden_layer_sizes=[10, 10], MLP_max_iter=1000,
                 DecisionTree_max_depth=5, SVC_kernel='linear', SVC_C=0.1):
        self.MLP_bagging = None
        self.DT_bagging = None
        self.SVC_bagging = None
        self.use_MLPClassifier = use_MLPClassifier
        self.use_DecisionTreeClassifier = use_DecisionTreeClassifier
        self.use_SVC =use_SVC
        self.soft_voting =soft_voting

        if use_MLPClassifier:
            self.MLP_bagging = BaggingClassifier(
                MLPClassifier(hidden_layer_sizes=MLP_hidden_layer_sizes, max_iter=MLP_max_iter),
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features
            )
        if use_DecisionTreeClassifier:
            self.DT_bagging = BaggingClassifier(
                DecisionTreeClassifier(max_depth=DecisionTree_max_depth),
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features
            )
        if use_SVC:
            self.SVC_bagging = BaggingClassifier(
                SVC(kernel=SVC_kernel, C=SVC_C),
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features
            )

    def fit(self, X_train, y_train):
        if self.use_MLPClassifier:
            self.MLP_bagging.fit(X_train, y_train)
        if self.use_DecisionTreeClassifier:
            self.DT_bagging.fit(X_train, y_train)
        if self.use_SVC:
            self.SVC_bagging.fit(X_train, y_train)

    def __call__(self, X):
        if self.soft_voting:
            # sum the prob, and argmax the result
            prob = 0
            if self.use_MLPClassifier:
                prob += self.MLP_bagging.predict_proba(X)
            if self.use_DecisionTreeClassifier:
                prob += self.DT_bagging.predict_proba(X)
            if self.use_SVC:
                prob += self.SVC_bagging.predict_proba(X)
            predictions = np.argmax(prob, axis=1)
            return predictions
        else:
            # select the most frequent prediction
            predictions = []
            if self.use_MLPClassifier:
                predictions.append(self.MLP_bagging.predict(X))
            if self.use_DecisionTreeClassifier:
                predictions.append(self.DT_bagging.predict(X))
            if self.use_SVC:
                predictions.append(self.SVC_bagging.predict(X))
            predictions = np.vstack(predictions)
            mode_result = mode(predictions, keepdims=False)
            return mode_result[0]

