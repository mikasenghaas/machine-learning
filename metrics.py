# script to store metrics to evalaute ml algorithms
import numpy as np
from icecream import ic
from sklearn import metrics
import statsmodels.api as sm

np.random.seed(1)

def rss(y, pred):
    return np.sum((y-pred)**2)

def rse(y, pred):
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

def tss(y):
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

def test():
    pred = np.arange(-50, 50) # y = x
    y = pred + np.random.uniform(-5, 5, size=len(pred)) # y = x + unif(-5, 5)


    print(f'RSS (Sklearn): {None}\nRSS (Own): {rss(y, pred)}\n')
    print(f'RSE (Sklearn): {None}\nRSE (Own): {rse(y, pred)}\n')
    print(f'R2 (Sklearn): {metrics.r2_score(y, pred)}\nR2 (Own): {r2_score(y, pred)}\n')


if __name__ == '__main__':
    test()


