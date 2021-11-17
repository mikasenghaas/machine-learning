"""
The :mod:`eduml.svm` module includes machine learning algorithms associated with  
the concept of support vector machines.
"""

from .hard_margin_svc import HardMarginSVC
from .soft_margin_svc import SoftMarginSVC
from .svm import SVM

__all__ = [
        'HardMarginSVC',
        'SoftMarginSVC',
        'SVM'
        ]
