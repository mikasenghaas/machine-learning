import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from icecream import ic

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
    p | int                 : Number of features in training set == must be 1
    X | np.array(n,p)       : 2D-Array of feature matrix
    Y | np.array(n,)        : 1D-Array of target vector

    Methods:
    ---
    .fit(X, y)              : Trains model given training split
    .predict(X)             : Prediction from trained KNN model
    """

    def __init__(self, K=5, algorithm='brute'):
        # data specific params
        self.X = None
        self.y = None
        self.n = None
        self.p = None

        # model parameters
        self.weights = None

    def fit(self, X, y, cross_validate=False):
        self.X = np.array(X).reshape(-1, 1)
        self.y = np.array(y).reshape(-1, 1)
        self.n, self.p = self.X.shape

        assert self.p == 1, 'Cannot perform simple linear regression on more than one feature'

        slope = np.sum((self.X - np.mean(self.X)) * (self.y - np.mean(self.y))) / np.sum((self.X - np.mean(X)) ** 2)
        bias =  np.mean(y) - slope * np.mean(self.X)

        self.w = np.array([bias, slope])


    def predict(self, X):
        X = np.vstack((np.ones(X.shape[0]), X)).T
        return SimpleLinearRegression.f(X, self.w)


    @staticmethod
    def f(X, w):
        return X @ w

    def __len__(self):
        return self.n


def main():
    X, y = load_diabetes(return_X_y=True)
    i = 2

    reg = SimpleLinearRegression()
    reg.fit(X[:,i], y)

    fig, ax = plt.subplots()
    ax.scatter(X[:,i], y, s=5)
    xs = np.linspace(min(X[:, i]), max(X[:, i]), 100)
    ax.plot(xs, reg.predict(xs), color='red')

    plt.show()

if __name__ == '__main__':
    main()
