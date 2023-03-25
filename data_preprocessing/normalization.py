from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler


from torchvision.transforms import Normalize


"""
example:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

StandardScaler: This scaler scales the features so that they have zero mean and unit variance. 
It does this by subtracting the mean of the feature and then dividing by the standard deviation of 
the feature. The purpose of this scaler is to make all the features have the same scale and to ensure 
that the features are centered around zero.

MinMaxScaler: This scaler scales the features to a specified range 
(default range is [0, 1]). It does this by subtracting the minimum value of the feature and then dividing by the range 
(maximum value - minimum value) of the feature. The purpose of this scaler is to make all the features lie in the same 
range and to prevent any feature from dominating the others.

MaxAbsScaler: This scaler scales the features so that they are in the range [-1, 1]. 
It does this by dividing each feature value by the maximum absolute value of the feature. 
The purpose of this scaler is to make all the features lie in the same range 
while preserving the sign of the values.

Normalizer: This scaler scales the features so that each sample (i.e., each row in the feature matrix) has a Euclidean 
norm of 1. This is done by dividing each feature value by the Euclidean norm of the sample to which it belongs. 
The purpose of this scaler is to ensure that the feature vectors have the same scale, regardless of their original magnitudes.

RobustScaler: This scaler scales the features using statistics that are robust to outliers. Specifically, 
it subtracts the median of the feature and then divides by the interquartile range (IQR) of the feature. 
The IQR is the range between the 25th and 75th percentile of the feature. 
The purpose of this scaler is to make the scaling more robust to the presence of outliers, 
which can cause problems with other scalers that use mean and variance-based normalization.

"""