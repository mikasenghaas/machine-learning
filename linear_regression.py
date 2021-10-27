import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from loss import mse, mae
from metrics import r2
from sklearn.metrics import r2_score
from icecream import ic
from tqdm import tqdm

class LinearRegression:
    """
    Implementation of a linear regression, where p quantitative and qualitative
    feature predict a single quantitative response. Optimisation through GD 
    algorithm

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
        self.X = None
        self.y = None
        self.n = None
        self.p = None

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

        self.y = np.array(y).reshape(-1, 1)
        self.n, self.p = self.X.shape

        self.epochs = epochs
        self.lr = lr


        self.w = np.random.rand(self.p + 1).reshape(-1, 1)
        X = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis=1)

        self.training_history = []

        for e in range(self.epochs):
            # update model params
            pred = LinearRegression.f(X, self.w).reshape(-1, 1)
            loss = self.loss(self.y, pred)
            self.training_history.append(loss)

            # update weights
            self.w -= self.lr * LinearRegression.g(X, self.y, pred)

            # print training updates
# if (e+1) % 10 == 0:
# print(f'--> Epoch {e+1}: Training Loss: {loss}')


    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return LinearRegression.f(X, self.w)

    @staticmethod
    def f(X, w):
        return X @ w

    @staticmethod
    def g(X, y, pred):
        return 2 / len(y) * X.T @ (pred - y)

    def __len__(self):
        return self.n


def main():
    X, y = load_diabetes(return_X_y=True)
    #X = X[:, :5]

    reg = LinearRegression()
    reg.fit(X, y, epochs=10000, lr=0.2)
    pred = reg.predict(X)

    print(r2_score(y, pred))
    print(reg.w)
    
    """
    fig, ax = plt.subplots()
    ax.scatter(X, y)
    xs = np.linspace(min(X), max(X), 100)
    xs_pred = reg.predict(xs) 
    ax.plot(xs, reg.predict(xs), color='red')
    plt.show()
    """

if __name__ == '__main__':
    main()
