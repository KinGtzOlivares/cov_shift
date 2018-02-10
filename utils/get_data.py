#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
#from sklearn import mixture

def gaussian_mixture(n, means, covs, probs):
    k, d = means.shape[0], means.shape[1]
    assert means.shape[0] == covs.shape[0]

    pops = np.random.choice(range(k), p=probs, size=n)
    samples = np.zeros(shape = (n,d))
    for idx, pop in np.ndenumerate(pops):
        sample = np.random.multivariate_normal(means[pop], covs[pop], 1)
        samples[idx] = sample
    return samples

def conditional_expectation(conditional_func, X):
    y = np.zeros(shape=(X.shape[0]))
    for idx in range(X.shape[0]):
        y[idx] = conditional_func(X[idx,:])
    return y

def toy_class(x):
    p = .5 * (1 + math.tanh(x[0]+min(0, x[1])))
    y = np.random.choice([0, 1], p = [p, 1-p])
    return y

def get_data_sugiyama(n_train, n_test):
    means_train = np.array([[-2, 3], [2, 3]])
    covs_train = np.array([[[1, 0],[0, 2]],[[1, 0],[0, 2]]])
    probs_train = np.array([.5, .5])
    
    means_test = np.array([[0, -1], [4, -1]])
    covs_test = np.array([[[1, 0],[0, 1]],[[1, 0],[0, 1]]])
    probs_test = np.array([.5, .5])
    
    X_train = gaussian_mixture(n_train, means_train, covs_train, probs_train)
    X_test = gaussian_mixture(n_test, means_test, covs_test, probs_test)
    
    y_train = conditional_expectation(toy_class, X_train)
    y_test = conditional_expectation(toy_class, X_test)

    return X_train, y_train, X_test, y_test


    