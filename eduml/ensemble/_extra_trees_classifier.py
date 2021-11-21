import numpy as np

from ._bagging_classifier import BaggingClassifier
from ..tree import DecisionTreeClassifier
from ..metrics import binary_gini, gini, entropy
from ..utils._validate import check_consistent_length, validate_feature_matrix, validate_target_vector

class ExtraTreesClassifier(BaggingClassifier):
    def __init__(self, 
                 criterion='gini', max_depth=None, max_features='auto',
                 number_of_trees=100, sample_size=1, bootstrap=True, random_state=0): 


        super().__init__(model=DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, algorithm='random', max_features='auto'), 
                         m=number_of_trees,
                         sample_size=sample_size,
                         bootstrap=bootstrap,
                         random_state=random_state)
