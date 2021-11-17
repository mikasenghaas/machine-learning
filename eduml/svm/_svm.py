import cvxopt
import cvxopt.solvers
import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec

from sklearn.svm import SVC
from sklearn.datasets import make_moons

class SVM():
    def __init__(self, kernel='linear', C=0, gamma=1, degree=3):
        if C is None:
            C=0
        if gamma is None:
            gamma = 1
        if kernel is None:
            kernel = 'linear'

        # hyperparameters
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.kernel = kernel


    def linear_kernel(self, x, y):
        """
        Linear Kernel is the simple dot-product of two vectors in same dimensionality
        """
        return x @ y

    def polynomial_kernel(self, x, y, C=1, degree=3):
        """
        Polynomial Kernel of arbitrary degree
        """
        return ((x @ y) + C) ** degree 

    def gaussian_kernel(self, x, y, gamma=0.5):
        """
        Gaussian Kernel
        """
        return np.exp(-gamma*linalg.norm(x - y) ** 2 )

    def fit(self, X, y):
        self.n, self.p = X.shape

        # Gram matrix
        K = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if self.kernel == 'linear':
                    K[i, j] = self.linear_kernel(X[i], X[j])
                if self.kernel=='gaussian':
                    K[i, j] = self.gaussian_kernel(X[i], X[j], self.gamma) 
                    self.C = None   # not used in gaussian kernel.
                if self.kernel == 'polynomial':
                    K[i, j] = self.polynomial_kernel(X[i], X[j], self.C, self.degree)


        # Converting into cvxopt format:
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(self.n) * -1)
        A = cvxopt.matrix(y, (1, self.n))
        b = cvxopt.matrix(0.0)

        if self.C is None or self.C==0:
            G = cvxopt.matrix(np.diag(np.ones(self.n) * -1))
            h = cvxopt.matrix(np.zeros(self.n))
        else:
            # Restricting the optimisation with parameter C.
            tmp1 = np.diag(np.ones(self.n) * -1)
            tmp2 = np.identity(self.n)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(self.n)
            tmp2 = np.ones(self.n) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # setting options for optimisation
        cvxopt.solvers.options['show_progress'] = True
        cvxopt.solvers.options['abstol'] = 1e-10
        cvxopt.solvers.options['reltol'] = 1e-10
        cvxopt.solvers.options['feastol'] = 1e-10

        # solve QP problem
        res = cvxopt.solvers.qp(P, q, G, h, A, b)

        # lagrange multipliers
        alphas = np.ravel(res['x'])   # flatten 2d output into a vector of lagrange multipliers

        # support vectors have non zero lagrange multipliers
        sv = alphas > 1e-5
        ind = np.arange(len(alphas))[sv]
        self.alphas = alphas[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # bias (For linear it is the intercept):
        self.b = 0
        for n in range(len(self.alphas)):
            # For all support vectors:
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alphas * self.sv_y * K[ind[n], sv])
        self.b = self.b / len(self.alphas)

        # weight vector
        if self.kernel == 'linear':
            self.w = np.zeros(self.n)
            for n in range(len(self.alphas)):
                self.w += self.alphas[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        # create the decision boundary for the plots. Calculates the hypothesis.
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.alphas, self.sv_y, self.sv):
                    # a : Lagrange multipliers, sv : support vectors.
                    # Hypothesis: sign(sum^S a * y * kernel + b)

                    if self.kernel == 'linear':
                        s += a * sv_y * self.linear_kernel(X[i], sv)
                    if self.kernel=='gaussian':
                        s += a * sv_y * self.gaussian_kernel(X[i], sv, self.gamma)   
                        self.C = None   # not used in gaussian kernel.
                    if self.kernel == 'polynomial':
                        s += a * sv_y * self.polynomial_kernel(X[i], sv, self.C, self.degree)

                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        # Hypothesis: sign(sum^S a * y * kernel + b).
        # NOTE: The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
        return np.sign(self.project(X))




if __name__ == '__main__':
    X, y = make_moons() 

    kernels = ['linear', 'polynomial', 'gaussian']
    cs = [0.5, 1, 10, 100]
    nrows = len(kernels)
    ncols = len(cs)

    gs = gridspec.GridSpec(nrows, ncols)
    figsize = (5*nrows, 3*ncols)
    fig = plt.figure(figsize=figsize)

    for i, kernel in zip(range(nrows), kernels):
        for j, c in zip(range(ncols), cs):
            clf = SVM(kernel=kernel, C=c)
            clf.fit(X, y)
            ax = plt.subplot(gs[i, j])
            fig = plot_decision_regions(X, y, clf, legend=2)
            plt.title(f'SVM with {kernel.title()} Kernel and C={c}')

    plt.show()
