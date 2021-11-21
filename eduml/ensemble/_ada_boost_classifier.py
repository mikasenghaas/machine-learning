import numpy as np 
from scipy.stats import mode

from ..tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from ..utils import *
from ..metrics import accuracy_score


class AdaBoostClassifier:
    """
    The AdaBoostClassifier() is a meta-level ensemble classifier, that sequentially transforms a single
    weak base learner (default: DecisionTreeClassifier(max_depth=1)) into a strong learner, by continuously 
    retrain a model, that focuses on correcting mistakes from previous estimator.

    Through its sequential training approach, AdaBoostClassifier() - just as any other boosting technique - 
    does not scale as well as bagging or pasting ensemble methods, and should therefore not be used on huge
    data.

    Running Time depends on number of estimators/ models to be trained (m) and the individual 
    training performance of the chosen model.

    All data specific attributes are intialised to 'None'
    and updated on call of .fit(). Transfer Training is not supported -
    a second call to .fit() will fully retrain the model.

    Attributes:
    ---
    n | int                 : Number of observed datapoints in training set
    p | int                 : Number of features in training set == must be 1
    X | np.array(n,p)       : 2D-Array of feature matrix
    Y | np.array(n,)        : 1D-Array of target vector

    model | estimator       : Base Estimator (usually weak learner)
    m | int                 : Number of Estimators

    Methods:
    ---
    .fit(X, y)              : Trains model given training split
    .predict(X)             : Predicts n data points 
    """
    def __init__(self, model=DecisionTreeClassifier(max_depth=1), m=10):
        # generic attributes
        self.X = self.y = self.n = self.p = None
        self.fitted = False

        # model specific attributes
        self.model = model
        self.m = m
        self.clfs = [self.model] * self.m
        self.model_weights = np.empty(self.m)
        self.sample_weights = None


    def fit(self, X, y):
        self.X = validate_feature_matrix(X)
        self.y = validate_target_vector(y)
        check_consistent_length(self.X, self.y)
        self.n, self.p = self.X.shape

        # initialise equal weights
        self.sample_weights = np.full(self.n, fill_value= 1/self.n)

        for i in range(self.m):
            # fit model 
            self.clfs[i].fit(self.X, self.y) # TODO: make .fit() fit with respect to sample weights

            # compute model weight
            preds = self.clfs[i].predict(self.X)
            model_weight = accuracy_score(self.y, preds)

            self.model_weights[i] = model_weight

            # recompute sample weights
            for j in range(self.n):
                if preds[j] != self.y[j]:
                    self.sample_weights[j] *= np.exp(model_weight)

        self.fitted = True

    def predict(self, X):
        X = validate_feature_matrix(X)
        N = X.shape[0]

        votes = np.array([model.predict(X).tolist() for model in self.clfs]).T
        preds = np.empty(N) 
        for i in range(N):
            weighted_votes = {c: 0 for c in np.unique(votes[i])}
            for j in range(self.m):
                weighted_votes[votes[i][j]] += float(self.model_weights[j])

            preds[i] = max(weighted_votes, key=weighted_votes.get)

        return preds
