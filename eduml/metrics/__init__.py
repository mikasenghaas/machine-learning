"""
The :mod:`eduml.metrics` module includes score functions, performance metrics
and pairwise metrics and distance computations.
"""

from ._classification import accuracy_score
from ._classification import classification_error
from ._classification import confusion_matrix
from ._classification import recall_score
from ._classification import precision_score
from ._classification import f1_score
from ._classification import classification_report

from ._regression import rss_score
from ._regression import rse_score
from ._regression import tss_score
from ._regression import r2_score
from ._regression import mean_squared_error

from ._loss import mse
from ._loss import se
from ._loss import mae 
from ._loss import zero_one_loss 
from ._loss import binary_cross_entropy 
from ._loss import cross_entropy 

from ._split import binary_gini
from ._split import gini
from ._split import entropy
from ._split import mse_split

__all__ = [
        'accuracy_score',
        'classification_error',
        'confusion_matrix',
        'recall_score',
        'precision_score',
        'f1_score',

        'rss_score',
        'rse_score',
        'tss_score', 
        'r2_score',

        'mse',
        'se',
        'mae',
        'zero_one_loss',
        'binary_cross_entropy',
        'cross_entropy'
        'binary_gini',
        'gini',
        'entropy',
        'mse_split'
        ]

