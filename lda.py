import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from mlxtend.plotting import plot_decision_regions
from icecream import ic
from tqdm import tqdm

class LDA:
    """
    Implementation of multivariate linear discriminant analysis (LDA), where p quantitative 
    features are used in a multiclass classification setting, ie. to distinguish between k distinct classes.

    LDA is a generative probabilistic model, that tries to estimate Bayes Classifier through indirectly estimating
    the posterior class probailities P(Y=k|X=x), by estimating the class conditonals P(X=x|Y=k) and prior P(X) and P(Y) and
    using Bayes Theorem to model P(Y=k|X=x). LDA makes the two simplifying assumptions that the class conditionals 
    are normally distributes and have common variance among classes (common covariance in case of p>1).

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

    def __init__(self):
        # data specific params
        self.X = None
        self.y = None
        self.n = None
        self.p = None

        # model parameters
        self.pi_ks = None 
        self.mu_ks = None
        self.cov = None

    def fit(self, X, y, cross_validate=False):
        self.X = np.array(X)
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)

        self.y = np.array(y).reshape(-1, 1)
        self.n, self.p = self.X.shape
        self.k = len(np.unique(self.y))

        # compute estimates to compute posterior class probabilities
        self.pi_ks = LDA.estimate_pi_k(y) # class prior estimates
        self.mu_ks = LDA.estimate_mu_k(X, y) # estimate of mean feature vector within each class
        self.cov = LDA.estimate_cov(X) # estimate of covariance matrix

    def predict(self, X):
        preds = []
        for k in range(self.k):
            mu_k = self.mu_ks[k] # mean vector for class k
            pi_k = self.pi_ks[k] # class prior of class k

            preds.append(LDA.d(X, self.cov, mu_k, pi_k))

        return np.argmax(np.array(preds), axis=0)

    def predict_proba(self, X):
        return np.empty(0)

    @staticmethod
    def d(X, cov, mu_k, pi_k):
        return X @ np.linalg.inv(cov) @ mu_k - (1 / 2 * mu_k @ np.linalg.inv(cov) @ mu_k) + np.log(pi_k)

    @staticmethod
    def estimate_pi_k(y):
        return np.unique(y, return_counts=True)[1] / y.shape[0]

    @staticmethod
    def estimate_mu_k(X, y):
        k = len(np.unique(y))
        mu_ks = []
        for i, k in enumerate(np.unique(y)):
            mu_ks.append(np.mean(X[y==k], axis=0))
        return np.array(mu_ks)

    @staticmethod
    def estimate_cov(X):
        return np.cov(X.T) # transpose because np.cov assumes to have features in rows, not cols

    def __len__(self):
        return self.n


def main():
    X, y = load_iris(return_X_y=True)
    X = X[:, :2]

    clf = LDA()
    clf.fit(X, y)

    # plot_decision_regions(X, y, clf, mesh_size=0.1)
    fig = plot_decision_regions(X, y, clf)
    plt.show()

if __name__ == '__main__':
    main()
