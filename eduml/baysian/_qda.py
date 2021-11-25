import numpy as np

from .._base import BaseClassifier

class QDA(BaseClassifier):
    """
    Implementation of multivariate quadratic discriminant analysis (LDA), where p quantitative 
    features are used in a multiclass classification setting, ie. to distinguish between k distinct classes.

    QDA is a generative probabilistic model, that tries to estimate Bayes Classifier through indirectly estimating
    the posterior class probailities P(Y=k|X=x), by estimating the class conditonals P(X=x|Y=k) and prior P(X) and P(Y) and
    using Bayes Theorem to model P(Y=k|X=x). In contrast to LDA, QDA relaxes the constraint that the covariance matrix is 
    constant among all classes. This relaxation leads to a discriminant funciton that is no longer linear, but quadratic.

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
        super().__init__()

        # model parameters
        self.pi_ks = None 
        self.mu_ks = None
        self.cov_ks = None

    def fit(self, X, y, cross_validate=False):
        self.X = np.array(X)
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)

        self.y = np.array(y).reshape(-1, 1)
        self.n, self.p = self.X.shape
        self.k = len(np.unique(self.y))

        # compute estimates to compute posterior class probabilities
        self.pi_ks = QDA.estimate_pi_k(y) # class prior estimates
        self.mu_ks = QDA.estimate_mu_k(X, y) # estimate of mean feature vector within each class
        self.cov_ks = QDA.estimate_cov_k(X, y) # estimate of covariance matrix

    def predict(self, X):
        preds = []
        for k in range(self.k):
            mu_k = self.mu_ks[k] # mean vector for class k
            pi_k = self.pi_ks[k] # class prior of class k
            cov_k = self.cov_ks[k]
            probs = []

            for x in X:
                probs.append(QDA.d(x, cov_k, mu_k, pi_k))
            preds.append(probs)

        return np.argmax(np.array(preds), axis=0)

    def predict_proba(self, X):
        return np.empty(0)

    @staticmethod
    def d(X, cov_k, mu_k, pi_k):
        return (-.5 * ((X-mu_k) @ np.linalg.inv(cov_k) @ (X-mu_k).T) - (.5 * np.log(np.linalg.det(cov_k))) + np.log(pi_k))

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
    def estimate_cov_k(X, y):
        k = len(np.unique(y))
        cov_ks = []
        for i, k in enumerate(np.unique(y)):
            cov_ks.append(np.cov(X[y==k].T))
        return cov_ks
