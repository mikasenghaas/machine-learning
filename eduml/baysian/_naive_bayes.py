import numpy as np
from scipy.stats import norm

class NaiveBayes:
    """
    Implementation of the popular Naive Bayes classifier, used to distinguish 
    k different classes from p quantiative features.
    Note that this implementation currently doesn't support qualitative 
    features, and will treat them exactly as the 
    quantitative features.

    Naive Bayes is a generative probabilistic model, that tries to estimate 
    Bayes Classifier through indirectly estimating
    the posterior class probailities P(Y=k|X=x), by estimating the class 
    conditonals P(X=x|Y=k) and prior P(X) and P(Y) and
    using Bayes Theorem to model P(Y=k|X=x). Naive Bayes assumes independence 
    of features among each class and models 
    the individual feature conditional distribution as univariate gaussian 
    density functions. 

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
        self.params = None

    def fit(self, X, y, cross_validate=False):
        self.X = np.array(X)
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)

        self.y = np.array(y).reshape(-1, 1)
        self.n, self.p = self.X.shape
        self.k = len(np.unique(self.y))

        # compute estimates to compute posterior class probabilities
        self.pi_ks = NaiveBayes.estimate_pi_k(y) # class prior estimates
        self.params = NaiveBayes.estimate_params(X, y) # estimate of mean and variance within each class for each feature (kxp list of list of tuples) 

    def predict(self, X):
        preds = []
        for k in range(self.k):
            params_k = self.params[k] # mean and variance for each feature p in the class k 
            pi_k = self.pi_ks[k] # class prior of class k

            preds.append(NaiveBayes.d(X, params_k, pi_k, self.k))

        return np.argmax(np.array(preds), axis=0)

    def predict_proba(self, X):
        preds = []
        for k in range(self.k):
            params_k = self.params[k] # mean and variance for each feature p in the class k 
            pi_k = self.pi_ks[k] # class prior of class k

            preds.append(NaiveBayes.d(X, params_k, pi_k, self.k))

        return np.max(np.array(preds), axis=0)

    @staticmethod
    def d(X, params_k, pi_k, k):
        return pi_k * np.prod([norm.pdf(X[:, p], params_k[p, 0], params_k[p, 1]) for p in range(X.shape[1])], axis=0) 
                 

    @staticmethod
    def estimate_pi_k(y):
        """
        Helperfunction to estimate the class prior for each class. Stored in a one-dimensional vector of length k.
        The value at index k represents the class prior pi_k for class k.
        """
        return np.unique(y, return_counts=True)[1] / len(y) 

    @staticmethod
    def estimate_params(X, y):
        """
        Helperfunction to estimate class-specfic mean and variance in each feature.
        Outputs a three-dimensional matrix in the shape k x p x 2 (mean and std are estimated)
        """
        return np.array([[(np.mean(X[y==k, p]), np.std(X[y==k, p])) for p in range(X.shape[1])] for k in np.unique(y)])

    def __len__(self):
        return self.n


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_iris
    from mlxtend.plotting import plot_decision_regions

    X, y = load_iris(return_X_y=True)
    X = X[:, :2]

    clf = NaiveBayes()
    clf.fit(X, y)

    # plot_decision_regions(X, y, clf, mesh_size=0.1)
    fig = plot_decision_regions(X, y, clf)
    plt.show()
