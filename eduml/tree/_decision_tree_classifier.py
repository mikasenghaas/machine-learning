import math
from collections import Counter
import numpy as np 

from ._decision_tree import DecisionTree

class DecisionTreeClassifier(DecisionTree):
    def __init__(self, criterion='gini', max_depth=None):

        super().__init__(max_depth=max_depth)

        if criterion == 'gini':
            self.criterion = DecisionTreeClassifier.gini
        elif criterion == 'binary_gini':
            self.criterion = DecisionTreeClassifier.binary_gini
        elif criterion == 'entropy':
            self.criterion = DecisionTreeClassifier.entropy
        else: 
            raise Exception('Cannot find this criterion')


    def _evaluate_leaf(self, node):
        labels = self.y[node.values]
        counter = Counter(labels)
        most_frequent_class = counter.most_common()[0][0]

        return most_frequent_class

    def _best_split(self, X, y):
        min_impurity = 1
        best_pair = None 
        for p in range(self.p):
            sorted_vals = sorted(list(set(X[:, p])))
            splits = [(sorted_vals[i]+sorted_vals[i+1]) / 2 for i in range(len(sorted_vals)-1)]
            for val in splits: 
                lower_val = X[:, p] < val
                split1 = y[lower_val]
                split2 = y[~lower_val]

                impurity1 = self.criterion(split1)
                impurity2 = self.criterion(split2)

                weighted_impurity = (impurity1 * len(split1) + impurity2 * len(split2)) / self.n
                # print(weighted_gini)

                if weighted_impurity < min_impurity:
                    min_impurity = weighted_impurity
                    best_pair = (p, val)

        return min_impurity, best_pair

    # impurity measures (classification tree loss-metrics)
    @staticmethod
    def binary_gini(y):
        p = len(y[y==0]) / len(y)
        return 2 * p * ( 1 - p ) 

    @staticmethod
    def gini(y):
        N = len(y)
        counter = Counter(y)

        ans = 0
        for val in counter.values():
            ans += val / N * ( 1 - val / N )
        return ans

    @staticmethod 
    def entropy(y):
        N = len(y)
        counter = Counter(y)

        ans = 0
        for val in counter.values():
            ans += val / N * np.log(val / N)
        return -ans


if __name__ == '__main__':
    cols = np.array(['red', 'blue'])
    X, y = datasets.load_iris(return_X_y=True)
    X = X[y!=2, :2]
    y = y[y!=2]
    
    #y = np.where(y==1, 0, 1) 

    clf = DecisionTreeClassifier(max_depth=None, criterion='gini')
    print(clf)
    clf.fit(X, y)
    print(clf)
    print(len(clf))
    print(clf.num_leaf_nodes)

    pred = clf.predict(X)
    print(accuracy_score(y, pred))

    fig = plot_decision_regions(X, y, clf)

    plt.show()
