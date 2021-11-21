import numpy as np

from ..utils import validate_feature_matrix, validate_target_vector, check_consistent_length

class SimpleLinearRegression:
    """
    Implementation of a simple linear regression, where a single quantitative
    feature predicts a single quantitative response. Based on derivative of
    MSE (Mean Squared Error)-Loss to find the two best-fitting model parameters
    beta_0 and beta_1

    Training Running Time: O(1)
    Training Space Complexity: O(1)

    Predict Running Time: O(1)
    Predict Space Complexity: O(1)


    All data specific attributes are intialised to 'None'
    and updated on call of .fit(). Transfer Training is not supported - 
    a second call to .fit() will fully retrain the model.

    Attributes:
    ---
    n | int                 : Number of observed datapoints in training set
    p | int                 : Number of features in training set; must be 1
    X | np.array(n,p)       : 2D-Array of feature matrix
    Y | np.array(n,)        : 1D-Array of target vector

    Methods:
    ---
    .fit(X, y)              : Trains model given training split
    .predict(X)             : Prediction from trained KNN model
    """

    def __init__(self):
        # generic attribute 
        self.X = self.y = self.n = self.p = None
        self.fitted = None

        # model parameters
        self.slope = None
        self.bias = None

    def fit(self, X, y):
        self.X = validate_feature_matrix(X)
        #self.X = self.X.reshape(-1)
        self.y = validate_target_vector(y)
        check_consistent_length(self.X, self.y)
        self.n, self.p = self.X.shape

        assert self.p == 1, 'Cannot perform simple linear regression on more than one feature'
        
        mean_x = np.mean(self.X)
        mean_y = np.mean(self.y)
        
        self.slope = np.sum( (self.X.reshape(-1) - mean_x) * (self.y - mean_y) ) / np.sum((self.X - mean_x) ** 2)
        self.bias =  mean_y - self.slope * mean_x

        self.fitted = True

    def predict(self, X):
        X = validate_feature_matrix(X)
        return (X * self.slope) + self.bias

    def __len__(self):
        return self.n
