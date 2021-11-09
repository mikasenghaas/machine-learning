# access parent dir
import sys
import os
sys.path.append("..") # Adds higher directory to python modules path.

# global imports
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from tqdm import tqdm

# custom imports 
from multinomial_logistic_regression import MultinomialLogisticRegression
from lda import LDA
from qda import QDA
from naive_bayes import NaiveBayes
from knn import KNN
from hard_margin_svc import HardMarginSVC
from plotting import plot_decision_regions

SHOW_FIGURES = False
SAVE_FIGURES = True

names = ['LogisticRegresion', 'LDA', 'QDA', 'NaiveBayes', 'KNN', 'HardMarginSVC']
clfs = [MultinomialLogisticRegression(), LDA(), QDA(), NaiveBayes(), KNN(), HardMarginSVC()]
N = len(clfs)

X, y = load_iris(return_X_y = True)
X = X[y!=2, :2]
y = y[y!=2]
# y = np.where(y[y!=2]==0, -1, y[y!=2])

for i, name, clf in tqdm(list(zip(list(range(N)), names, clfs))):
    clf.fit(X, y)

    #fig, ax = plt.subplots()
    #ax.scatter(X[:, 0], X[:, 1], c=y)
    fig = plot_decision_regions(X, y, clf, title=f'{name}')

    if SHOW_FIGURES:
        plt.show()

    if SAVE_FIGURES:
        os.mkdir('plotting') if not os.path.exists('plotting') else None
        fig.savefig(f'plotting/{name}.jpg')
