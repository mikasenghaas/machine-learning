import numpy as np
from matplotlib import pyplot as plt
from loss import mse
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from icecream import ic

class PolynomialRegression:
    """
    Implementation of a polynomial regression, where a set of quantitative (or qualitative)
    feature predicts a single quantitative response.
    Here, we assume the model to be a nth-degree polynomial (by default: 2nd degree polynomial (quadratic formula)). 
    Model learns through vanilla GD.

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
    degrees | int           : Number of Degrees the fitted model function has

    Methods:
    ---
    .fit(X, y)              : Trains model given training split
    .predict(X)             : Prediction from trained KNN model
    """

    def __init__(self, loss=mse, degrees=2):
        # data specific params
        self.X = None
        self.y = None
        self.n = None
        self.p = None

        # training parameters
        self.epochs = None
        self.lr = None
        self.loss = loss 
        self.training_history = []

        # model parameters
        self.degrees = degrees
        self.weights = None

    def fit(self, X, y, cross_validate=False, epochs=1000, lr=0.01):
        self.X = np.array(X).reshape(-1, 1)
        self.y = np.array(y).reshape(-1, 1)
        self.n, self.p = self.X.shape

        self.weights = np.random.rand(self.degrees + 1).reshape(-1, 1)
        X = SimplePolynomialRegression.expand_feature_matrix(self.X, self.degrees)

        self.epochs = epochs 
        self.lr = lr

        for e in range(self.epochs):
            pred = SimplePolynomialRegression.f(X, self.weights).reshape(-1, 1)

            loss = self.loss(self.y, pred)  
            self.training_history.append(loss)

            self.weights -= self.lr * SimplePolynomialRegression.g(X, self.y, pred)


    def predict(self, X):
        X = SimplePolynomialRegression.expand_feature_matrix(X, self.degrees)
        return SimplePolynomialRegression.f(X, self.weights)


    @staticmethod
    def f(X, w):
        return X @ w

    @staticmethod
    def g(X, y, pred):
        return 2 / len(y) * X.T @ (pred - y)

    @staticmethod
    def expand_feature_matrix(X, degrees):
        for p in range(X.shape[1]):
            for d in range(1, degrees):
                X = np.concatenate((X, (X[:, p] ** (d+1)).reshape(-1,1)), axis=1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return X


    def __len__(self):
        return self.n


def main():
    X = np.random.uniform(-10, 10, size=(100, 1))
    y = X ** 3 - X ** 2 + 2*X + 2


    # X, y = load_diabetes(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    reg = SimplePolynomialRegression(degrees=3)
    reg.fit(X, y, epochs=1000, lr=0.01)


    fig, ax = plt.subplots(ncols=2, figsize=(8, 3))
    ax[0].scatter(X, y, s=5)
    xs = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    ax[0].plot(xs, reg.predict(xs), color='red')
    ax[0].set_title('Fitted Model on Data')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    
    ax[1].plot(np.arange(len(reg.training_history)), reg.training_history)
    ax[1].set_title('Training History (Loss: MSE)')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Training Loss (MSE)')


    plt.show()

if __name__ == '__main__':
    main()

