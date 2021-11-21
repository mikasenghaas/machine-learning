import math
import numpy as np 

from ._decision_tree import DecisionTree
from ..metrics import mse_split

class DecisionTreeRegressor(DecisionTree):
    def __init__(self, criterion='mse', 
                 max_depth=None,
                 algorithm='greedy',
                 k=None):

        super().__init__(max_depth=max_depth, algorithm=algorithm, k=k)

        if criterion == 'mse':
            self.criterion = mse_split 
        else: 
            raise Exception('Cannot find this criterion')

    def _evaluate_leaf(self, node):
        vals = self.y[node.values]

        return np.mean(vals)
