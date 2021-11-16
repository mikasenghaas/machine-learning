import math
from collections import Counter
import numpy as np 
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from icecream import ic

from sklearn import datasets
from sklearn import tree
from sklearn.metrics import *

class Node:
    def __init__(self, size=None, values=None, depth=None, _type='internal'):
        self.size = size # number of samples to be split in the node
        self.depth = depth # depth of node in decision tree
        self.values = values # indices of the sample to be split in the node

        self.p = None # feature id 
        self.val = None # value to split at (default decision True, if data point lower than val -> in tree vis lower right branch)
        self.gini = None
        self.decision = lambda x: int(x[self.p] < self.val)  # lambda function 
        self.split = [None, None] # amount of samples in split1 and split2values
        self.next = [None, None] # reference to new splits
        self.type = _type

    def __str__(self):
        return f'Decision Rule for {self.type.title()} Node at Depth {self.depth} (Gini: {round(self.gini, 2) if self.gini!=None else None}): X[{self.p}]<{self.val}; Splitting {self.size} values into {self.split}'

class DecisionTree:
    def __init__(self, max_depth=None):
        self.X = None
        self.y = None

        # decision tree metrics
        self.root = None
        if max_depth == None:
            self.max_depth = math.inf
        self.max_depth = max_depth

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n, self.p = self.X.shape

        # root node
        self.root = Node(size=self.n, values=np.arange(self.n), depth=1)

        self._split(self.root)

    def predict(self, X):
        preds = []
        for x in X:
            #print('predict: ', x)
            curr = self.root
            #print(curr)
            while curr.gini != 0 and curr.depth < self.max_depth:
                curr = curr.next[curr.decision(x)]
                #print(curr)

            pred = self._evaluate_leaf(curr, x)

            preds.append(pred)

        return np.array(preds)


    def _split(self, curr):
        # curr is initialised as node with size, indices of values and depth
        # find best split
        X, y = self.X[curr.values], self.y[curr.values] # consider training samples that are in split of current node

        gini, best_pair = self._best_split(X, y) # find best pair to split further

        # assign gini and split criterion
        p, val = best_pair
        curr.gini = gini
        curr.p = p 
        curr.val = val

        # compute new split
        train_decisions = []
        for x in X:
            train_decisions.append(curr.decision(x))
        train_decisions = np.array(train_decisions)

        curr.split = [curr.size - sum(train_decisions), sum(train_decisions)]

        print(curr)

        # find new indices in splits
        next_values = [[], []]
        for i in curr.values:
            if curr.decision(self.X[i]) == 0:
                next_values[0].append(i)
            else:
                next_values[1].append(i)

        #next_values = [np.array(next_values[0]), np.array(next_values[1])]

        # initialise new nodes
        curr.next[0] = Node(size=curr.split[0], values=next_values[0], depth=curr.depth+1)

        if DecisionTree.binary_gini(self.y[curr.next[0].values]) != 0 and curr.depth < self.max_depth:
            self._split(curr.next[0])
        else:
            curr.next[0].type = 'leaf'

        curr.next[1] = Node(size=curr.split[1], values=next_values[1], depth=curr.depth+1)

        if DecisionTree.binary_gini(self.y[curr.next[1].values]) != 0 and curr.depth < self.max_depth:
            self._split(curr.next[1])
        else:
            curr.next[1].type = 'leaf'


    def _best_split(self, X, y):
        best_gini = 1
        best_pair = None
        for p in range(self.p):
            sorted_vals = sorted(list(set(X[:, p])))
            splits = [(sorted_vals[i]+sorted_vals[i+1]) / 2 for i in range(len(sorted_vals)-1)]
            for val in splits: 
                lower_val = X[:, p] < val
                split1 = y[lower_val]
                split2 = y[~lower_val]
                # print(len(split1))
                # print(len(split2))

                gini1 = DecisionTree.binary_gini(split1)
                gini2 = DecisionTree.binary_gini(split2)
                # print('gini1 ', gini1)
                # print('gini2 ', gini2)

                weighted_gini = (gini1 * len(split1) + gini2 * len(split2)) / self.n
                # print(weighted_gini)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_pair = (p, val)

        return best_gini, best_pair
    
    def _evaluate_leaf(self, node, x):
        if node.type == 'internal':
            decision = node.decision(x)
            #print(bool(decision))
            final_region = node.next[decision]

            #print(final_region.values)
            labels = self.y[final_region.values]
            #print(labels)
            counter = Counter(labels)
            #print(counter)

            most_frequent_class = counter.most_common()[0][0]
            #print('prediction: ', most_frequent_class)
        elif node.type == 'leaf':
            labels = self.y[node.values]
            counter = Counter(labels)
            most_frequent_class = counter.most_common()[0][0]
            

        return most_frequent_class


    @staticmethod
    def binary_gini(y):
        p = len(y[y==0]) / len(y)
        return 2 * p * ( 1 - p ) 



if __name__ == '__main__':
    cols = np.array(['red', 'blue'])
    X, y = datasets.load_iris(return_X_y=True)
    X = X[y!=2, :2]
    y = y[y!=2]
    
    #y = np.where(y==1, 0, 1) 

    clf = DecisionTree(max_depth=3)
    clf.fit(X, y)
    #test = np.array([[0,0], [6.5, 2.5]])

    pred = clf.predict(X)
    print(pred)
    print(accuracy_score(y, pred))


    """
    fig, ax = plt.subplots()
    ax.scatter(X[y==0, 0], X[y==0, 1], c='red', label='Class 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Class 1')
    ax.legend(loc='best')
    #ax.scatter(test[:, 0], test[:, 1], )
    """

    fig = plot_decision_regions(X, y, clf)

    plt.show()
