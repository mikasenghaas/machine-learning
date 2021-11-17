"""
The :mod:`eduml.baysian` module includes three classical statistical learning methods
that are based on the ideal Bayes Classifier. 
"""

from ._lda import LDA
from ._qda import QDA
from ._naive_bayes import NaiveBayes

__all__ = [
        'LDA',
        'QDA',
        'NaiveBayes'
        ]

