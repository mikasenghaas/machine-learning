import math
import numpy as np 

from ._decision_tree import DecisionTree

class DecisionTreeRegressor(DecisionTree):
    def __init__(self, criterion='mse', 
                 max_depth=None):

        super().__init__(max_depth=max_depth)

        if criterion == 'mse':
            self.criterion = DecisionTreeRegressor.mse
        else: 
            raise Exception('Cannot find this criterion')

    def _evaluate_leaf(self, node):
        vals = self.y[node.values]

        return np.mean(vals)

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        preds = []
        for x in X:
            curr = self.root
            while not curr.is_leaf():
                if curr.decision(x) == True:
                    curr = curr.right
                else: 
                    curr = curr.left
            preds.append(curr.prediction)

        return np.array(preds)

    def _best_split(self, X, y):
        min_loss = math.inf 
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

                if weighted_impurity < min_loss:
                    min_loss = weighted_impurity
                    best_pair = (p, val)

        return min_loss, best_pair

    # impurity measures (loss-metrics)
    @staticmethod
    def mse(y):
        # compute average of y vec
        y_mean = np.mean(y)
        return np.mean(np.sum((y-y_mean)**2))

if __name__ == '__main__':
    X = np.random.uniform(low=-5, high=5, size=100)
    y_generate = lambda x: x**2 + 2 + np.random.normal(scale=2)

    y = np.array(list(map(y_generate, X)))

    reg = DecisionTreeRegressor(max_depth=3)
    reg.fit(X, y)
    print(reg)

    fig, ax = plt.subplots()
    ax.scatter(X, y)

    xs = np.linspace(-5, 5, 100)
    ys = reg.predict(xs)
    ax.plot(xs, ys, c='red')

    plt.show()
    
