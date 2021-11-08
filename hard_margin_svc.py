import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from icecream import ic


class HardMarginSVC:
    """
    The Hard Margin SVC (also: Maximal Margin Classifier) tries to find a hyperplane in p-1 dimensional space
    that linearly separates two distinct classes. It can thus be used for high dimensional
    feature matrices in the binary classification setting, if there exist a sound linear
    decision boundary.
    The algorithm is not recommended for use, but should be seen as an educational towards
    the well-generalising SVM (support vector machine) classifier.

    All data specific attributes are intialised to 'None'
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
    .fit(X, y)              : Trains model given training split
    .predict(X)             : Prediction from trained KNN model
    """

    def __init__(self, kernel='linear'):
        # data specific params
        self.X = None
        self.y = None
        self.n = None
        self.p = None

        self.kernel = kernel
        self.support_idx = None
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n, self.p = self.X.shape

        # find support vectors
        K = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if self.kernel == 'linear':
                    K[i][j] = HardMarginSVC.linear_kernel(self.X[i], self.X[j])
                else:
                    print('Kernel has not been implemented yet.')

        def loss(a, *args):
            """Evaluate the negative of the dual function at `a`.
            We access the optimization data (Gram matrix K and target vector y) from outer scope for convenience.
            :param a: dual variables (alpha)
            """
            a = a.reshape(1,-1)   # reshape a as we assumed it as a 1 x N matrix in the equations above
            yv = y.reshape(-1,1)  # reshape y as we assumed it as a N x 1 matrix in the equations above
            A = (a.T@a) * (yv@yv.T) * K
            
            return - (np.sum(a) - (.5 * np.sum(A))) # might be only the sum over the diagnoal (not sure about the notation in the sum)

        def jac(a, *args):
            """Calculate the Jacobian of the loss function (for the QP solver)"""
            a = a.reshape(1,-1)
            yv = y.reshape(-1,1)
            j = - np.ones_like(a) + a @ ((yv @ yv.T) * K)

            return j.flatten()

        A = np.eye(self.n)  
        a0 = np.random.rand(self.n)  # initial guess for alpha vector (randomised
        y = self.y 
        constraints = ({'type': 'ineq', 'fun': lambda a: A @ a, 'jac': lambda a: A},
                       {'type': 'eq', 'fun': lambda a: a @ y.T, 'jac': lambda a: y.T})

        a = minimize(loss, a0, jac=jac, args=(y, K), constraints=constraints, method='SLSQP', options={}).x
        a[np.isclose(a, 0)] = 0 # zero out all non-support vectors

        self.support_idx = np.where(a > 0)[0] 

        X_sv = self.X[self.support_idx]
        y_sv = self.y[self.support_idx]
        a_sv = a[self.support_idx]

        self.b = HardMarginSVC.compute_b(X_sv, y_sv, a_sv, HardMarginSVC.linear_kernel)
        self.w = HardMarginSVC.compute_weights(self.p, X_sv, y_sv, a_sv, HardMarginSVC.linear_kernel)

    def predict(self, X):
        return np.where(self.predict_val(X), 1, -1)  

    def predict_val(self, X):
        return X @ self.w + self.b

    def predict_proba(self, X):
        pass

    # kernels
    @staticmethod
    def linear_kernel(x, y):
        return x @ y

    # compute model params
    @staticmethod
    def compute_weights(p, X_sv, y_sv, a_sv, kernel=None):
        w = np.zeros(p)
        N = len(a_sv)
        for i in range(N):
            w += a_sv[i] * y_sv[i] * X_sv[i]
        return w

    @staticmethod
    def compute_b(X_sv, y_sv, a_sv, kernel):
        # input only consist of support vectors (X: support vector features, y: support vector labels, a: ...)
        N = len(a_sv)
        b = 0
        for i in range(N):
            b += y_sv[i]
            b -= np.sum(a_sv * y_sv * kernel(X_sv, X_sv[i]))
        return b / N

    def __len__(self):
        return self.n


def main():
    SHOW_FIGURES = True

    X, y = load_iris(return_X_y=True)
    exclude_2 = y != 2
    X = X[exclude_2, :2]
    y = y[exclude_2]
    y = np.where(y==0, -1, y)
    
    clf = HardMarginSVC(kernel='linear')
    clf.fit(X, y)

    sv_ids = clf.support_idx

    # fig = plot_decision_regions(X, y, clf)
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=np.array([None, 'red', 'blue'])[y])
    ax.scatter(X[sv_ids, 0], X[sv_ids, 1], facecolors='None', edgecolors='black', linewidth=2, label='Support Vectors')

    grid1 = np.arange(X[:, 0].min()*0.9, X[:, 0].max()*1.1, 0.01)
    grid2 = np.arange(X[:, 1].min()*0.9, X[:, 1].max()*1.1, 0.01)
    xx, yy = np.meshgrid(grid1, grid2)
    zs = clf.predict_val(np.array(list(zip(np.ravel(xx), np.ravel(yy)))))
    zz = zs.reshape(xx.shape)
    ax.contour(xx, yy, zz, levels=[-1, 0, 1], colors='black', linestyles=['dotted', 'solid', 'dotted'], linewidth=1) 

    # ax.clabel(CS, fmt='%2.1d', colors='k')
    ax.set_title('Hard Margin SVC Classification on Iris Data')
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    ax.legend(loc='best')


    if SHOW_FIGURES:
        plt.show()

if __name__ == '__main__':
    main()

