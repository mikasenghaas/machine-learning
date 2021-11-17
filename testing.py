from eduml.linear_model import LogisticRegression
from eduml.plotting import plot_decision_regions
#from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris 
from matplotlib import pyplot as plt

X, y = load_iris(return_X_y=True)
X = X[:, :2]

clf = LDA()
clf.fit(X, y, epochs=100000)

fig = plot_decision_regions(X, y, clf)
plt.show()
