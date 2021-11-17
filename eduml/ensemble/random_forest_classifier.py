from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.metrics import *
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier as bench
from tqdm import tqdm

from decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, number_of_trees=100, random_state=0):
        np.random.seed(random_state)
        self.X = None
        self.y = None

        self.number_of_trees = number_of_trees
        self.clfs = [DecisionTreeClassifier(max_depth=None)] * self.number_of_trees

    def fit(self, X, y):
        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1)
        else:
            self.X = np.array(X)
        self.y = np.array(y)
        self.n, self.p = self.X.shape

        for i in range(self.number_of_trees):
            # bootstrap data 
            idx = self._generate_bootstrap(n=self.n)
            X, y = self.X[idx], self.y[idx]

            self.clfs[i].fit(X, y) 

    def predict(self, X):
        preds = []
        for clf in self.clfs:
            preds.append(clf.predict(X))
        preds = np.array(preds).T
       
        ans = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=preds)

        return ans

    def _generate_bootstrap(self, n):
        return [np.random.randint(self.n) for _ in range(n)]


if __name__ == '__main__':
    X, y = datasets.load_iris(return_X_y=True)
    X = X[y!=3, :2]
    y = y[y!=3]
    
    #y = np.where(y==1, 0, 1) 

    clf = RandomForestClassifier(number_of_trees=15)
    #clf = bench()
    clf.fit(X, y)

    pred = clf.predict(X)
    print(accuracy_score(y, pred))

    fig = plot_decision_regions(X, y, clf)

    plt.show()



