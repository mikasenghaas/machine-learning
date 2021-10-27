import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from loss import mse, se, mae, zero_one_loss

class MLMethod:
    """
    Skeleton class that is being used to implement different 
    machine learning algorithms. Class is initialised with the method's
    hyperparameters. All data specific attributes are intialised to 'None'
    and updated on call of .fit(). Transfer Training is not supported - 
    a second call to .fit() will fully retrain the model.

    Attributes:
    ---
    n | int                 : Number of observed datapoints in training set
    p | int                 : Number of features in training set
    X | np.array(n,p)       : 2D-Array of feature matrix
    Y | np.array(n,)        : 1D-Array of target vector

    Methods:
    ---
    .fit(X, y)              : Trains model given training split; 



    """
    def __init__(self, loss=mse):
        # data specific params
        self.X = None
        self.y = None
        self.n = None
        self.p = None

        # model specific params
        self.f = self.X @ self.w
        self.loss = loss 

        # hyperparameters

    def fit(self, X, y, cross_validate=False):
        self.X = np.array(X)
        self.y = np.array(y).reshape(-1, 1)
        self.n, self.p = self.X.shape


    def predict(self, X):
        pass

    def plot(self):
        assert self.X not None, 'Please train the model'
        assert self.p == 2, 'Plotting only supported for p=2 features'
        fig, ax = plt.subplots(ncols=2, figsize=(16, 6))

        ax[0] = plot_decision_regions(X=X_train, y=y_train.astype(int),
                                    clf=model, legend=2)
        ax[1] = plot_decision_regions(X=X_train, y=y_train.astype(int),
                                    clf=model, legend=2)
        plt.show()

    def __len__(self):
        return self.n
