import numpy as np 
from scipy.stats import mode

from sklearn.tree import DecisionTreeClassifier # replace with eduml at some point
from sklearn.metrics import accuracy_score

from ..utils import *


class VotingClassifier:
    """
    The VotingClassifier() is a meta-level ensemble classifier, that trains m hetereogeneouos classifiers
    on the training data. The introduced variation in the training of the models promises to result in 
    better-generalising models. 

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

    clfs* | iterable        : List of Classifier  

    Methods:
    ---
    .fit(X, y)              : Trains model given training split
    .predict(X)             : Predicts n data points 
    """
    def __init__(self, *clfs, voting='hard'):
        # generic attributes
        self.X = self.y = self.n = self.p = None
        self.fitted = False

        # model specific attributes
        self.clfs = clfs
        self.m = len(self.clfs)
        self.voting = voting


    def fit(self, X, y):
        self.X = validate_feature_matrix(X)
        self.y = validate_target_vector(y)
        check_consistent_length(self.X, self.y)
        self.n, self.p = self.X.shape

        for i in range(self.m):
            # train each classifier on entire dataset
            self.clfs[i].fit(self.X, self.y)
        
        self.fitted = True

    def predict(self, X):
        X = validate_feature_matrix(X)
        N = X.shape[0]

        if self.voting == 'hard':
            votes = np.array([model.predict(X).tolist() for model in self.clfs])
            preds = np.array(mode(votes, axis=0)[0][0])
        elif self.voting == 'soft':
            soft_votes = np.array([model.predict_proba(X).tolist() for model in self.clfs])
            preds = []
            for i in range(N):
                pred = np.argmax(np.sum(soft_votes[:, i, :], axis=0) / N)
                preds.append(pred)
            preds = np.array(preds)

        return preds
