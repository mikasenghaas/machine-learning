"""
The :file:`eduml.metrics._classification` module includes score functions, performance 
metrics for classification problems.
"""

import numpy as np
from ..utils import check_consistent_length

def accuracy_score(y_true, y_pred, normalised=True):
    #check_consistent_length(y_true, y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    correct = sum(y_true == y_pred)

    if normalised:
        return correct / len(y_true)
    return correct

def misclassification(y_true, y_pred, normalised=True):
    #check_consistent_length(y_true, y_pred)

    misclassified = sum(y_true == y_pred)

    if normalised:
        return misclassified / len(y_true)
    return misclassified

if __name__ == '__main__':
    print(accuracy_score([1, 2, 3], [1, 2, 3]))
