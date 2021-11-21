"""
The :mod:`eduml.ensemble` module includes functionality to construct ensemble
statistical learning methods to produce better generalising models.
"""

from ._bagging_classifier import BaggingClassifier
from ._voting_classifier import VotingClassifier
from ._random_forest_classifier import RandomForestClassifier
from ._extra_trees_classifier import ExtraTreesClassifier

__all__ = [
        'RandomForestClassifier',
        'BaggingClassifier',
        'VotingClassifier',
        'ExtraTreesClassifier'
        ]
