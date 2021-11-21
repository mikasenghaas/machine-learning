import numpy as np
from math import inf

from ..metrics import mse  
from ..utils import validate_feature_matrix, validate_target_vector, check_consistent_length

class LinearRegression:
    """
    Implementation of a linear regression, where p quantitative and qualitative
    feature predict a single quantitative response. Optimisation until gradient
    descent converges or fixed number of epochs is reached.

    Running Time depends on number of features (p), epochs (e):

    Training Running Time: O(e * )
    Training Space Complexity: O(1)

    Predict Running Time: O(1)
    Predict Space Complexity: O(1)


    All data specific attributes are intialised to 'None'
    and updated on call of .fit(). Transfer Training is not supported - 
    a second call to .fit() will fully retrain the model.

    Attributes:
    ---
    n | int                 : Number of observed datapoints in training set
    p | int                 : Number of features in training set == must be 1
    X | np.array(n,p)       : 2D-Array of feature matrix
    Y | np.array(n,)        : 1D-Array of target vector

    Methods:
    ---
    .fit(X, y)              : Trains model given training split
    .predict(X)             : Prediction from trained KNN model
    """

    def __init__(self, loss=mse, optim='GD'):
        # data specific params
        self.X = self.y = self.n = self.p = None
        self.fitted = False

        # gradient descent training
        self.optim = optim
        self.loss = loss
        self.epochs = None 
        self.lr = None
        self.training_history = []

        # model parameters
        self.weights = None
        self.bias = None

    def fit(self, X, y, epochs=None, lr=0.01, verbose=False):
        self.X = validate_feature_matrix(X)
        self.y = validate_target_vector(y)
        check_consistent_length(self.X, self.y)
        self.n, self.p = self.X.shape

        self.epochs = epochs if epochs != None else inf
        self.lr = lr

        self.weights = np.random.rand(self.p)
        self.bias = np.random.rand()

        e = 0
        while True:
            # update model params
            e += 1
            pred = self.predict(self.X)
            loss = self.loss(self.y, pred)
            self.training_history.append(loss)

            # update weights
            self.weights -= self.lr * self._gradient_weights(pred)
            self.bias -= self.lr * self._gradient_bias(pred)

            if e > 1:
                improvement = np.abs(self.training_history[-1] - self.training_history[-2]) 
            else:
                improvement = 1

            # print training updates
            if verbose:
                if (e+1) % 50 == 0:
                    print(f'Epoch {e+1}: Training Loss: {loss}, Improvement: {improvement}')

            # stop criterion
            if improvement < 0.0001 or e >= self.epochs:
                break

        self.fitted = True

    def predict(self, X):
        X = validate_feature_matrix(X)
        return X @ self.weights + self.bias

    def _gradient_weights(self, pred):
        return 2 / len(self.y) * self.X.T @ (pred - self.y)

    def _gradient_bias(self, pred):
        return 2 / len(self.y) * np.sum(pred - self.y)

    def __len__(self):
        return self.n
