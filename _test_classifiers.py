import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# EDUML
from eduml.linear_model import BinaryLogisticRegression
from eduml.linear_model import LogisticRegression

from eduml.baysian import LDA
from eduml.baysian import QDA
from eduml.baysian import NaiveBayes 

from eduml.neighbors import KNN

from eduml.svm import HardMarginSVC

from eduml.tree import DecisionTreeClassifier

from eduml.ensemble import RandomForestClassifier
from eduml.ensemble import ExtraTreesClassifier
from eduml.ensemble import AdaBoostClassifier
from eduml.ensemble import BaggingClassifier
from eduml.ensemble import VotingClassifier

from eduml.plotting import plot_2d_decision_regions

X, y = load_iris(return_X_y=True)
X = X[y!=2, :2]
y = y[y!=2]

clfs = [BinaryLogisticRegression, LogisticRegression, LDA, QDA, NaiveBayes, 
        DecisionTreeClassifier, KNN, HardMarginSVC]

names = ['Binary Logistic Regression', 'Logistic Regression', 'LDA', 'QDA', 'Naive Bayes',
         'Decision Tree', 'KNN', 'Extra Trees Classifier', 'Hard Margin SVC']


fig, axs = plt.subplots(nrows=1, ncols=len(clfs), squeeze=False, figsize=(3*len(clfs), 3))

for ax, clf, name in zip(axs.ravel(), clfs, names):
    try: 
        model = clf()
        model.fit(X,y)
        plot_2d_decision_regions(X, y, model, meshsize=0.01, title=name, ax=ax)

    except Exception as e:
        print(f"{name} failed because of '{e}'")


plt.show()
