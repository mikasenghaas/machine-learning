# script that contains commonly used loss-function in machine learning
import numpy as np

def mse(y, p):
    return 1 / len(y) * sum((y - p)**2)

def se(y, p):
    return sum((y - p)**2)

def mae(y, p):
    return 1 / len(y) * sum(np.abs(y-p))

def zero_one_loss(y, p):
    return np.sum(y != p)

def binary_cross_entropy(y, p):
    return - (1 / len(y)) * np.sum(y * np.log(p) + ((1+y) * np.log(1-p)))

def cross_entropy(y, p):
    return - (1 / y.shape[0]) * np.sum(y * np.log(p))
