"""
The :mod:`eduml.metrics` module includes score functions, performance metrics
and pairwise metrics and distance computations.
"""

from ._classification import accuracy_score
from ._classification import misclassification

from ._loss import mse
from ._loss import se
from ._loss import mae 
from ._loss import zero_one_loss 
from ._loss import binary_cross_entropy 
from ._loss import cross_entropy 

__all__ = [
        'accuracy_score',
        'misclassification',

        'mse',
        'se',
        'mae',
        'zero_one_loss',
        'binary_cross_entropy',
        'cross_entropy'
        ]
