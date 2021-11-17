"""
The :mod:`eduml.tree` module includes tree-based machine learning algorithms.
"""

from ._decision_tree import DecisionTree
from ._decision_tree_classifier import DecisionTreeClassifier
from ._decision_tree_regressor import DecisionTreeRegressor

__all__ = [
        'DecisionTree',
        'DecisionTreeClassifier',
        'DecisionTreeRegressor'
        ]

