"""
The :mod:`eduml.linear_model` module includes statistical learning methods that
are based on the linear model.
"""

from ._simple_linear_regression import SimpleLinearRegression
from ._linear_regression import LinearRegression
from ._binary_logistic_regression import BinaryLogisticRegression
from ._logistic_regression import LogisticRegression

__all__ = [
        'SimpleLinearRegression',
        'LinearRegression',
        'BinaryLogisticRegression',
        'LogisticRegression'
        ]

