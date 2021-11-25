import numpy as np
from sklearn.neighbors import BallTree
from scipy.stats import mode

from .._base import BaseClassifier


class KNN(BaseClassifier):
    """
    KNN classifier implementation supporing different algorithms for computation 
    of k-nearest neighbours:

    'brute-force'
    ------------
    For each unseen training datapoint computes the distance to all samples in 
    training set and finds k closest. assigns to class that appears most often
    in the neighbourhood set.

    Training Time: O(1)
    Training Space: O(1)
    Prediction Time: O(k * n * p) // for each observation compute distance and 
                                  // find the k nearest (call to min() k times)
    Prediction Space: O(1) 

    'ball-tree'
    -----------
    The ball-tree algorithm initialises a ball-tree data structure during .fit()
    This increases the training time significantly, but allows for faster 
    prediction.

    Training Time: O(n * p * log(n))
    Training Space: O(n * p)
    Prediction Time: O(k * log(n))
    Prediction Space: O(1)

    => NOT IMPLEMENTED YET

    Class is initialised with the method's hyperparameters, here:
    self.K (int)    : The number of k neigbours to compute estimate 
                      of posterior class condtionals
    self.algorithm (str)  : Algorithm to use

    All data specific attributes are intialised to 'None'
    and updated on call of .fit(). Transfer Training is not supported - 
    a second call to .fit() will fully retrain the model.

    Attributes:
    ---
    n | int                 : Number of observed datapoints in training set
    p | int                 : Number of features in training set
    X | np.array(n,p)       : 2D-Array of feature matrix
    Y | np.array(n,)        : 1D-Array of target vector

    Methods:
    ---
    .fit(X, y)              : Trains model given training split
    .predict(X)             : Prediction from trained KNN model
    """

    def __init__(self, K=5, algorithm='brute'):
        # data specific params
        super().__init__()

        # hyperparameters
        self.K = K
        self.algorithm = algorithm

    def fit(self, X, y, cross_validate=False):
        self.X = np.array(X)
        self.y = np.array(y).reshape(-1, 1)
        self.n, self.p = self.X.shape

        if self.algorithm == 'ball-tree':
            # initialise ball tree
            self.tree = BallTree(self.X)


    def predict(self, X):
        if self.algorithm == 'brute':
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            k_nearest = np.array([sorted([(KNN.euclidean_distance(self.X[i], x), 
                        self.y[i].item()) for i in range(self.n)], 
                        key=lambda x:x[0]) for x in X])[:, :self.K, 1]

            return (mode(k_nearest, axis=1)[0]).reshape(-1,)


        elif self.algorithm == 'ball-tree':
            pass

    def predict_proba(self, X):
        if self.algorithm == 'brute':
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            k_nearest = np.array([sorted([(KNN.euclidean_distance(self.X[i], x), 
                        self.y[i].item()) for i in range(self.n)], 
                        key=lambda x:x[0]) for x in X])[:, :self.K, 1]

            return (mode(k_nearest, axis=1)[1] / self.K).reshape(-1,)

        elif self.algorithm == 'ball-tree':
            pass
            

    @staticmethod
    def euclidean_distance(x, y):
        return np.sqrt(np.sum((x-y)**2))
