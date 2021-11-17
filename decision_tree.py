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
        self.loss = None
        self.decision = lambda x: x[self.p] < self.val  # lambda function 

        self.split = [None, None] # amount of samples in split1 and split2values
        self.left = None # if decision evaluates False
        self.right = None # if decision evaluates True

        # leaf node
        self.type = _type
        self.prediction = None

    def is_leaf(self):
        return self.type == 'leaf'

    def __str__(self):
        if self.type == 'root':
            return 'Root Node at Depth 1 '\
                   f'(Loss: {round(self.loss, 2) if self.loss!=None else None}): '\
                   f'X[{self.p}] < {self.val}; Splitting {self.size} values in ' \
                   f'[False=={self.split[0]}, True=={self.split[1]}]'

        elif self.type == 'internal':
            return f'Internal Node at Depth {self.depth} ' \
                   f'(Loss: {round(self.loss, 2) if self.loss!=None else None}): '\
                   f'X[{self.p}] < {self.val}; Splitting {self.size} values in '\
                   f'[False=={self.split[0]}, True=={self.split[1]}]'

        elif self.type == 'leaf':
            return f'Leaf Node at Depth {self.depth} '\
                   f'(Loss: {round(self.loss, 2) if self.loss!=None else None}): '\
                   f'Prediction: {self.prediction}'



class DecisionTree:
    """
    Parent Class,not intended for use. Use children classes 
    - DecisionTreeClassifier 
    - DecisionTreeRegressor
    """
    def __init__(self, max_depth=None, max_nodes=None, max_leaf_nodes=None, min_nodes_per_leaf=1):

        # training data
        self.X = None
        self.y = None

        # decision tree metrics
        self.root = None
        self.num_nodes = 0
        self.num_leaf_nodes = 0

        self.criterion = None

        # stopping criterion
        if max_depth == None: self.max_depth = math.inf 
        else: self.max_depth = max_depth

        if max_nodes == None: self.max_nodes = math.inf
        else: self.max_nodes = max_nodes

        if max_leaf_nodes == None: self.max_leaf_nodes = math.inf
        else: self.max_leaf_nodes = max_leaf_nodes

        self.min_nodes_per_leaf = min_nodes_per_leaf

    def fit(self, X, y):
        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1)
        else:
            self.X = np.array(X)
        self.y = np.array(y)
        self.n, self.p = self.X.shape

        # root node
        self.root = Node(size=self.n, values=np.arange(self.n), depth=1, _type='root')
        self.num_nodes += 1

        self._split(self.root)

    def predict(self, X):
        preds = []
        for x in X:
            #print('predict: ', x)
            curr = self.root
            #print(curr)
            while not curr.is_leaf():
                if curr.decision(x) == True:
                    curr = curr.right
                else: 
                    curr = curr.left

            preds.append(curr.prediction)

        return np.array(preds)

    def predict_proba(self, X):
        # undefined for single decision tree
        return np.empty(0)

    def __len__(self):
        return self.num_nodes

    def __str__(self):
        if self.num_nodes > 0:
            curr = self.root
            q = []
            q.insert(0, curr)
            depth = 0
            s = ''

            while q != []:
                curr = q.pop()
                if curr.depth > depth:
                    depth = curr.depth
                    s += f'\nCurrent Depth: {curr.depth}'
                    s += f"{'='*15}\n"
                s += curr.__str__() + '\n'

                if curr.left != None:
                    q.insert(0, curr.left)
                if curr.right != None:
                    q.insert(0, curr.right)
        else: 
            s = 'Decision Tree is not fitted'

        return s

    def _is_pure(self, node):
        return self.criterion(self.y[node.values]) == 0

    def _check_criterion(self, node):
        depth_not_reached = node.depth < self.max_depth
        max_nodes_not_reached = self.num_nodes < self.max_nodes
        max_leaf_nodes_not_reached = self.num_leaf_nodes < self.max_leaf_nodes 
        can_split = node.size > 1

        return (depth_not_reached and 
                max_nodes_not_reached and 
                max_leaf_nodes_not_reached and 
                can_split)

    def _split(self, curr):
        # curr is initialised as node with size, indices of values and depth
        # find best split
        X, y = self.X[curr.values], self.y[curr.values] # consider training samples that are in split of current node

        if self._best_split(X, y)[1] == None:
            # exception if there does not exist a good split
            # stop splitting and make node leaf
            curr.type = 'leaf'
            curr.prediction = self._evaluate_leaf(curr)
            self.num_leaf_nodes += 1
            return

        loss, best_pair = self._best_split(X, y) # find best pair to split further

        # assign gini and split criterion
        p, val = best_pair
        curr.loss = loss
        curr.p = p 
        curr.val = val

        print(curr)

        # compute new split
        train_decisions = []
        for x in X:
            train_decisions.append(curr.decision(x))
        train_decisions = np.array(train_decisions)

        curr.split = [curr.size - sum(train_decisions), sum(train_decisions)]

        # find new indices in splits
        next_values = [[], []]
        for i in curr.values:
            if curr.decision(self.X[i]) == 0:
                next_values[0].append(i)
            else:
                next_values[1].append(i)

        #next_values = [np.array(next_values[0]), np.array(next_values[1])]

        # initialise new nodes
        curr.left = Node(size=curr.split[0], values=next_values[0], depth=curr.depth+1)
        self.num_nodes += 1

        # split further if not pure or pre-pruning stop criterion not reached
        if not self._is_pure(curr.left) and self._check_criterion(curr.left):
            self._split(curr.left)
        else:
            # otherwise make leaf
            curr.left.type = 'leaf'
            curr.left.prediction = self._evaluate_leaf(curr.left)
            self.num_leaf_nodes += 1

        curr.right = Node(size=curr.split[1], values=next_values[1], depth=curr.depth+1)
        self.num_nodes += 1

        if not self._is_pure(curr.right) and self._check_criterion(curr.right):
            self._split(curr.right)
        else:
            curr.right.type = 'leaf'
            curr.right.prediction = self._evaluate_leaf(curr.right)
            self.num_leaf_nodes += 1


    
    """
    def _evaluate_leaf(self, node):
        labels = self.y[node.values]
        counter = Counter(labels)
        most_frequent_class = counter.most_common()[0][0]

        return most_frequent_class


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
    """


if __name__ == '__main__':
    cols = np.array(['red', 'blue'])
    X, y = datasets.load_iris(return_X_y=True)
    X = X[y!=2, :2]
    y = y[y!=2]
    
    #y = np.where(y==1, 0, 1) 

    clf = DecisionTree(max_depth=None, criterion='gini')
    print(clf)
    clf.fit(X, y)
    print(clf)
    print(len(clf))
    print(clf.num_leaf_nodes)

    pred = clf.predict(X)
    print(accuracy_score(y, pred))

    fig = plot_decision_regions(X, y, clf)

    plt.show()
