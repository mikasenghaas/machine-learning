import numpy as np
from math import inf

from ..metrics import mse, cross_entropy
from ..utils import validate_feature_matrix, validate_target_vector, check_consistent_length

class BinaryLogisticRegression:
    """
    Implementation of a simple logistic regression, where p quantitative and qualitative
    features are used in binary classification setting to distinguish between k=2 distinct classes.
    In logistic regression the logistic function f(x) = 1 / (1 + e^(-x)) is used to model the conditional 
    probability P(Y|X) directly. The logistic function is optimised using GD on call to .fit(). The method
    converges by default, by the training loop can manually be adjusted through the epochs and learning rate
    arguments.

    Running Time depends on number of features (p) and number of epochs (e):

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

    def __init__(self, db=0.5, loss=cross_entropy, optim='GD'):
        # generic attributes
        self.X = self.y = self.n = self.p = self.k = None
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
        self.db = db


    def fit(self, X, y, epochs=10000, lr=0.01, verbose=False):
        self.X = validate_feature_matrix(X)
        self.y = validate_target_vector(y)
        check_consistent_length(self.X, self.y)
        self.n, self.p = self.X.shape
        self.k = len(np.unique(self.y))

        assert self.k == 2, f'Simple Logistic Regression only supports binary classification.'\
                             'Use  LogisticRegression for multiclass classification.'

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
            if e >= self.epochs:
                break

        self.fitted = True


    def predict(self, X):
        return (self.predict_proba(X) > self.db).astype(int)

    def predict_proba(self, X):
        X = validate_feature_matrix(X)

        z = X @ self.weights + self.bias

        return 1 / (1 + np.exp(-z))

    def _gradient_weights(self, pred):
        return 2 / len(self.y) * self.X.T @ (pred - self.y)

    def _gradient_bias(self, pred):
        return 2 / len(self.y) * np.sum(pred - self.y)

    def __len__(self):
        return self.n
