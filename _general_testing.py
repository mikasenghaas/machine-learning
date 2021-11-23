import numpy as np

# EDUML
from eduml.linear_model import SimpleLinearRegression
from eduml.linear_model import LinearRegression
from eduml.linear_model import BinaryLogisticRegression
from eduml.linear_model import LogisticRegression

from eduml.baysian import LDA
from eduml.baysian import QDA
from eduml.baysian import NaiveBayes

from eduml.svm import HardMarginSVC

from eduml.tree import DecisionTreeClassifier
from eduml.tree import DecisionTreeRegressor

from eduml.ensemble import RandomForestClassifier
from eduml.ensemble import ExtraTreesClassifier
from eduml.ensemble import AdaBoostClassifier
from eduml.ensemble import BaggingClassifier
from eduml.ensemble import VotingClassifier

#from eduml.plotting import plot_1d_decision_regions
from eduml.plotting import plot_1d_regression
from eduml.plotting import plot_2d_decision_regions

from eduml.metrics import accuracy_score
from eduml.metrics import classification_error
from eduml.metrics import confusion_matrix
from eduml.metrics import precision_score
from eduml.metrics import recall_score
from eduml.metrics import f1_score
from eduml.metrics import classification_report

# SKLEARN
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, make_classification, make_regression

from sklearn.linear_model import LogisticRegression as benchmark
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix as conf_matrix
from sklearn.metrics import recall_score as rs
from sklearn.metrics import precision_score as ps
from sklearn.metrics import precision_score as f1 
from sklearn.metrics import classification_report as cr
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC

from time import time
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions

#np.random.seed(1)

X, y = load_iris(return_X_y = True)
#X, y = make_regression(100, 1, n_informative=1, noise=3)
X = X[y!=2, :2]
y = y[y!=2]


clf = NaiveBayes()

clf.fit(X, y)
plot_2d_decision_regions(X, y, clf)
plt.show()


#reg = LinearRegression()
#reg.fit(X_train, y_train, epochs=None, lr=0.0001, verbose=True)
#preds = reg.predict(X_train)


"""
for clf, name in zip([clf1, clf2], ['eduml', 'sklearn']):
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    #print('Training Acc ', accuracy_score(y_train, train_preds))
    #print('Test Acc ', accuracy_score(y_test, test_preds))
    #plot_1d_regression(X_train, y_train, clf, title=name)
    plot_2d_decision_regions(X_train, y_train, clf, title=name)
    plt.show()
"""
