"""
The :mod:`eduml.svm` module includes machine learning algorithms associated with  
the concept of support vector machines.
"""

from ._hard_margin_svc import HardMarginSVC
from ._soft_margin_svc import SoftMarginSVC
from ._svm import SVM

__all__ = [
        'HardMarginSVC',
        'SoftMarginSVC',
        'SVM'
        ]
