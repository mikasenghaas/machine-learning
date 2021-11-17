"""
The :mod:`eduml.ensemble` module includes functionality to construct ensemble
statistical learning methods to produce better generalising models.
"""

from ._random_forest_classifier import RandomForestClassifier

__all__ = [
        'RandomForestClassifier'
        ]
