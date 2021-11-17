import numpy as np
from matplotlib import pyplot as plt
from ..metrics import mse, binary_cross_entropy, cross_entropy

class LogisticRegression:
    """
    Implementation of multinomial logistic regression, where p quantitative and qualitative
    features are used in multiclass classification setting, ie. to distinguish between k>=2 distinct classes.
    In logistic regression the logistic function f(x) = 1 / (1 + e^(-x)) is used to model the conditional 
    probability P(Y|X) directly. The logistic function is optimised using vanilla GD.

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

    def __init__(self, loss=cross_entropy, optim='GD'):
        # data specific params
        self.X = None
        self.y = None
        self.n = None
        self.p = None
        self.k = None

        # training
        self.optim = optim
        self.loss = loss
        self.epochs = None 
        self.lr = None

        # model parameters
        self.weights = None

    def fit(self, X, y, cross_validate=False, epochs=500, lr=0.01):
        self.X = np.array(X)
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)

        self.y = np.array(y).reshape(-1,)
        self.n, self.p = self.X.shape
        self.k = len(np.unique(self.y))

        self.epochs = epochs
        self.lr = lr

        self.w = np.random.rand(self.p + 1, self.k) # weights matrix
        X = LogisticRegression.add_bias(self.X) # add bias to feature matrix
        y = np.eye(self.k)[self.y] # one hot encoded target vector y 

        self.training_history = []
        for e in range(self.epochs):
            # update model params
            pred = LogisticRegression.f(X, self.w)
            loss = self.loss(y, pred)
            self.training_history.append(loss)

            # update weights
            self.w -= self.lr * LogisticRegression.g(X, y, pred)


    def predict(self, X):
        X = LogisticRegression.add_bias(X)
        return np.argmax(LogisticRegression.f(X, self.w), axis=1)

    def predict_proba(self, X):
        X = LogisticRegression.add_bias(X)
        return np.max(LogisticRegression.f(X, self.w), axis=1)

    @staticmethod
    def f(X, w):
        # softmax
        z = X @ w
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)


    @staticmethod
    def g(X, y, pred):
        return 2 / len(y) * X.T @ (pred - y)

    @staticmethod
    def add_bias(X):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def __len__(self):
        return self.n


def main():
    from sklearn.datasets import load_iris
    from mlxtend.plotting import plot_decision_regions
    # k=3 classification
    X, y = load_iris(return_X_y=True)
    X = X[:, :2]

    clf = LogisticRegression()
    clf.fit(X, y, epochs=10000, lr=0.1)

    #plot_decision_regions(X, y, clf, mesh_size=0.1)
    fig = plot_decision_regions(X, y, clf)
    plt.show()

if __name__ == '__main__':
    main()
