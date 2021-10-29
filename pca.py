import numpy as np 
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from icecream import ic

class PCA:
    def __init__(self, num_principle_components=1):
        self.X = None
        self.n = None
        self.p = None

        # pca specific attributes
        self.num_of_components = num_principle_components
        self.cov = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, X):
        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1)
        else:
            self.X = X
        self.n, self.p = X.shape

        X_meaned = self.X - np.mean(self.X, axis=0)
        
        self.cov = np.cov(X_meaned, rowvar=False)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.cov)

        sorted_index = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[sorted_index]
        self.eigenvectors = self.eigenvectors[sorted_index]

    def transform(self, X):
        assert X.shape[1] == self.p, 'Dimensionality does not match'

        n_eigenvectors = self.eigenvectors[:,:self.num_of_components]

        X_meaned = X - np.mean(X, axis=0)
        self.X = (n_eigenvectors.T @ X_meaned.T).T

        return self.X


    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def main():
    X, y = datasets.load_iris(return_X_y=True)

    pca = PCA(3)
    pca_X = pca.fit_transform(X)

    clf = GaussianNB()
    clf.fit(X, y)
    print(f'Accuracy Score in four features: {accuracy_score(y, clf.predict(X))}')
    clf2 = GaussianNB()
    clf2.fit(pca_X, y)
    print(f'Accuracy Score in two principal components: {accuracy_score(y, clf2.predict(pca_X))}')





if __name__ == '__main__':
    main()
