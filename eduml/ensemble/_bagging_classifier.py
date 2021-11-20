import numpy as np 
from scipy.stats import mode

from sklearn.tree import DecisionTreeClassifier # replace with eduml at some point
from sklearn.metrics import accuracy_score

from ..utils import *


class BaggingClassifier:
    """
    The BaggingClassifier() is a meta-level ensemble classifier, that trains m equivalent classifier
    on different - randomised - subsets of the training data. The introduced randomness creates variation
    in the training of the models, which promises to result in better-generalising models.
    The bootstrap class argument determines, whether the randomly sampled subset for training the 
    individual estimators should sample with or without replacement. If bootstrap=True, then it is sampled
    with replacement (resulting in a bagging classifier), otherwise the method is known as pasting.
    
    The procedure of bagging is generally used to boost performance for a model that is already known to perform well as well as to 
    simply reduce overfitting. The final model predicts through a majority vote.

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

    m | int                 : Number of Estimators
    sample_size | float     : Determines the size of the sampled training batch
    bootstrap | bool        : 'True' if randomised  
    oob | bool              : If 'True' measures Out-Of-Bag Evaluation as Estimator for Out-Of-Sample Performance

    Methods:
    ---
    .fit(X, y)              : Trains model given training split
    .predict(X)             : Predicts n data points 
    """
    def __init__(self, model=DecisionTreeClassifier(), m=10, sample_size=1, bootstrap=True, oob=False, random_state=0):
        # generic attributes
        self.X = self.y = self.n = self.p = None

        # model specific attributes
        self.model = model
        self.m = m
        self.sample_size = sample_size
        self.clfs = [None] * self.m
        self.bootstrap = bootstrap 
        if oob:
            self.compute_oob = True
            self.oob = 0
        else:
            self.compute_oob = False
            self.oob = None

        # set random state
        np.random.seed(random_state)
        self.fitted = False

    def fit(self, X, y):
        self.X = validate_feature_matrix(X)
        self.y = validate_target_vector(y)
        check_consistent_length(self.X, self.y)
        self.n, self.p = self.X.shape

        for i in range(self.m):
            idx = np.random.choice(self.n, int(self.n*self.sample_size), replace=self.bootstrap) # sampling indices to create data splits
            X, y = self.X[idx], self.y[idx] # use sampled indices to mask data split

            # train classifier on data split
            self.clfs[i] = self.model.fit(X, y)

            if self.compute_oob: 
                oob_idx = list(set(range(self.n)) - set(idx))
                oob_preds = self.clfs[i].predict(X[oob_idx])
                oob_acc = accuracy_score(self.y[oob_idx], oob_preds)
                self.oob += accuracy_score(self.y[oob_idx], oob_preds)
        
        if self.compute_oob:
            self.oob /= self.m
        self.fitted = True

    def predict(self, X):
        X = validate_feature_matrix(X)

        votes = np.array([model.predict(X).tolist() for model in self.clfs])
        preds = np.array(mode(votes, axis=0)[0][0])

        return preds
