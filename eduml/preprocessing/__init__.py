"""
The :mod:`eduml.preprocessing` module includes common methods of preprocessing data
before fitting to a classifier.
"""

from .pca import PCA
from .transform import center
from .transform import scale
from .transform import standardise

__all__ = [
        'PCA',
        'center',
        'scale', 
        'standardise'
        ]
