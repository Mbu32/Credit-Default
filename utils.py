import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import FunctionTransformer

def winsorize_fn(X):
    return np.array(winsorize(np.array(X), limits=[0.01, 0.01], axis=0))

def make_winsorizer():
    return FunctionTransformer(winsorize_fn, feature_names_out='one-to-one')

def make_ratio(X):
    eps = 0.001
    return (X[:, [0]] / (X[:, [1]] + eps))

def monthlycash(X):
    return ((X[:, [0]] / 12) * (1 - (X[:, [1]] / 100)))

def ratio_name(function_transformer, feature_names_in):
    return ['custom_ratio']

