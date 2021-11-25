import numpy as np
import statsmodels.api as sm

from ..utils import check_consistent_length

def rss_score(y, pred):
    check_consistent_length(y, pred)
    y = y.flatten()
    pred = pred.flatten()

    return np.sum((y-pred)**2)

def mean_squared_error(y, pred):
    return rss_score(y, pred) / len(y)

def rse_score(y, pred):
    """
    Residual Sum of Errors (RSE) is a estimate of the standard deviation 
    of the error term in a regression fit. Used to evaluate the performance
    of a regression.

    Input Parameters
    ----------------
    y (np.ndarray)              : Vector of dimension nx1 that holds ground truth 
    pred (np.ndarray)           : Vector of dimnesion nx1 that holds predictions of
                                  ground truth

    Return
    ------
    rse (float)                 : Computed RSE-Score

    """
    return np.sqrt(1 / (len(y) -2) * rss(y, pred) )

def tss_score(y):
    return ((y - np.mean(y))**2).sum()

def r2_score(y, pred):
    """
    R2-Score (coefficient of determination score) to evaluate 
    predictions in a regression setting.

    Input Parameters
    ----------------
    y (np.ndarray)              : Vector of dimension nx1 that holds ground truth 
    pred (np.ndarray)           : Vector of dimnesion nx1 that holds predictions of
                                  ground truth

    Return
    ------
    r2_score (float)            : Computed R2-Score
    """
    return  1 - ( rse(y, pred) / tss(y) )
