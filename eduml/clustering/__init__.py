"""
The :mod:`eduml.clustering` module includes unsupervised learning methods for 
clustering p-dimensional data. 
"""

from ._k_means import KMeans
from ._hierarchical_clustering import HierarchicalClustering

__all__ = [
        'KMeans',
        'HierarchicalClustering'
        ]

