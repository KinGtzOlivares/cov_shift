#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
from scipy.optimize import linprog

import shutil
import os

def gaussian_mixture(n, means, covs, probs):
    k, d = means.shape[0], means.shape[1]
    assert means.shape[0] == covs.shape[0]

    pops = np.random.choice(range(k), p=probs, size=n)
    samples = np.zeros(shape = (n,d))
    for idx, pop in np.ndenumerate(pops):
        sample = np.random.multivariate_normal(means[pop], covs[pop], 1)
        samples[idx] = sample
    return samples

def conditional_sugiyama(X):
    y = np.zeros(shape=(X.shape[0]))
    for idx in range(X.shape[0]):
        x = X[idx,:]
        p = .5 * (1 + math.tanh(x[0]+min(0, x[1])))
        y[idx] = np.random.choice([0, 1], p = [p, 1-p])
    return y

def get_data_sugiyama(n_train, n_test):
    means_train = np.array([[-2, 3], [2, 3]])
    covs_train = np.array([[[1, 0],[0, 2]],[[1, 0],[0, 2]]])
    label_probs_train = np.array([.5, .5])
    
    means_test = np.array([[0, -1], [4, -1]])
    covs_test = np.array([[[1, 0],[0, 1]],[[1, 0],[0, 1]]])
    label_probs_test = np.array([.5, .5])
    
    X_train = gaussian_mixture(n_train, means_train, covs_train, label_probs_train)
    X_test = gaussian_mixture(n_test, means_test, covs_test, label_probs_test)
    
    y_train = conditional_sugiyama(X_train)
    y_test = conditional_sugiyama(X_test)

    return X_train, y_train, X_test, y_test

def in_hull(V, x):
    n_clusters, d = V.shape
    x = np.reshape(x, (1, d))
    c = np.zeros(n_clusters)
    A = np.concatenate(( V.T, np.ones((1, n_clusters)) ), axis=0)
    b = np.concatenate(( x.T, np.ones((1, 1)) ), axis=0)
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

def vertices_polygon(n_clusters, r):
    V = np.zeros(shape = (n_clusters, 2))
    for c in range(n_clusters):
        V[c,:] = [r * math.cos(2*math.pi*c/n_clusters), r * math.sin(2*math.pi*c/n_clusters)]
    return V

def conditional_polygon(X, n_clusters, r, n_grid):
    rs = np.array(np.linspace(1e-10, 10, n_grid))
    Vs = []
    for h, rd in np.ndenumerate(rs):
        V = vertices_polygon(n_clusters, rd)
        Vs.append(V)
    
    rds = []
    Y = []
    for c in range(n_clusters):
        X_c = X[c][0]
        nx = X_c.shape[0]
        rxs = np.zeros(shape = (nx, 1))
        ys = np.zeros(shape = (nx, 1))
        for n in range(nx):
            rxsup = 100
            for h, rd in np.ndenumerate(rs):
                if in_hull(Vs[h[0]], X_c[n]) == True:
                    rxsup = rd
                    break
            rxs[n] = rxsup
            p = .5 * (1 + math.tanh(.5 * (rxsup-r) )) # 100 *
            ys[n] = np.random.choice([0, 1], p = [p, 1-p])
        rds.append(rxs)
        Y.append(ys)
    return Y, rds

def get_population_parameters(n_clusters, r):
    mus = []
    covs = []
    V = vertices_polygon(n_clusters, r)
    for c in range(n_clusters):
        mu_0 = np.array([(4.0/10) * r * math.cos(2*math.pi*((c+.5)/n_clusters)), 
                    (4.0/10) * r * math.sin(2*math.pi*((c+.5)/n_clusters))])
        mu_1 = np.array([(12.0/10) * r * math.cos(2*math.pi*((c+.5)/n_clusters)), 
                    (12.0/10) * r * math.sin(2*math.pi*((c+.5)/n_clusters))])
        mus.append([mu_0, mu_1])
    
    for c in range(n_clusters):
        edge = V[int(math.fmod(c+1, n_clusters)),:]-V[c,:]
        edge = np.reshape(edge, (2, 1))
        mu_1 = mus[c][1]
        mu_1 = np.reshape(mu_0, (2,1))
        cov_0 = (1.0/16) * np.matmul(mu_1, mu_1.T) + (1.0/32) * np.matmul(edge, edge.T)
        cov_1 = (1.0/16) * np.matmul(mu_1, mu_1.T) + (1.0/32) * np.matmul(edge, edge.T)
        covs.append([cov_0, cov_1])
    return V, mus, covs

def get_data_experiment(n_samples, n_clusters, r, n_grid):
    V, mus, covs = get_population_parameters(n_clusters, r)
    c_probs = np.array([.5, .5])

    X = []
    n_cluster = n_samples/n_clusters
    X_c = np.zeros(shape = (n_cluster, 3))
    for c in range(n_clusters):
        X_c = gaussian_mixture(n_cluster, np.array(mus[c]), 
            np.array([covs[c][0], covs[c][1]]), c_probs)
        X_cluster = np.repeat(c, n_cluster)
        X.append([X_c, X_cluster])
    
    Y, _ = conditional_polygon(X, n_clusters, r, n_grid)

    return X, Y, V

def save_data_experiment(n_clusters, X, Y, V, path):
    shutil.rmtree(path)
    os.makedirs(path)

    np.save(file = (path + 'V'), arr = V)
    for c in range(n_clusters):
        X_c = np.column_stack((X[c][0],X[c][1])) 
        np.save(file = (path + 'X_') + str(c), arr = X_c)
        y_c = Y[c]
        np.save(file = (path + 'y_') + str(c), arr = y_c)

def load_data_experiment(n_clusters, path):
    X, Y = [], []
    V = np.load(file = (path + 'V.npy'))
    for c in range(n_clusters):
        X_l = np.load(file = (path + 'X_' + str(c) + '.npy'))
        X_c, X_cluster = X_l[:,0:2], X_l[:,2]
        X.append([X_c, X_cluster])
        y_c = np.load(file = (path + 'y_' + str(c) + '.npy'))
        Y.append(y_c)
    return X, Y, V

def split_data_experiment(n_clusters, X, Y, propscore_size = .1, test_size=.2):
    X_propscore, X_train, X_test = [], [], []
    Y_propscore, Y_train, Y_test = [], [], []
    for c in range(n_clusters):
        n_c = Y[c].shape[0]
        n_propscore, n_test = int(n_c * propscore_size), int(n_c * test_size)
        np.random.shuffle(X[c][0]); np.random.shuffle(X[c][1]); np.random.shuffle(Y[c])

        #X
        X_c, X_cluster = X[c][0], X[c][1]
        X_propscore_c = [ X_c[:n_propscore,:], X_cluster[:n_propscore] ]
        X_test_c = [ X_c[n_propscore:(n_propscore+n_test),:], X_cluster[n_propscore:(n_propscore+n_test)] ]
        X_train_c = [ X_c[(n_propscore+n_test):,:], X_cluster[(n_propscore+n_test):] ]
        X_propscore.append(X_propscore_c); X_train.append(X_train_c); X_test.append(X_test_c)

        #Y
        Y_propscore_c = Y[c][:n_propscore,:]
        Y_test_c = Y[c][n_propscore:(n_propscore+n_test),:]
        Y_train_c = Y[c][(n_propscore+n_test):,:]
        Y_propscore.append(Y_propscore_c); Y_train.append(Y_train_c); Y_test.append(Y_test_c)

    data = {'X_propscore': X_propscore, 'X_test': X_test, 'X_train': X_train, 
            'y_propscore': Y_propscore, 'y_test': Y_test, 'y_train': Y_train}
    return data

def parse_data_propscore(n_clusters, X_propscore, y_propscore):
    X = np.vstack([np.array(X_propscore[c][0]) for c in range(n_clusters)])
    y = np.concatenate([np.array(y_propscore[c]).flatten() for c in range(n_clusters)])
    
    clusters = np.concatenate([np.array(X_propscore[c][1]).flatten() for c in range(n_clusters)])
    proxy_cluster = (clusters + y*n_clusters)
    proxy_cluster = proxy_cluster.astype(int)
    return X, y, proxy_cluster

def parse_data(n_clusters, X_propscore, y_propscore):
    X = np.vstack([np.array(X_propscore[c][0]) for c in range(n_clusters)])
    y = np.concatenate([np.array(y_propscore[c]).flatten() for c in range(n_clusters)])
    
    clusters = np.concatenate([np.array(X_propscore[c][1]).flatten() for c in range(n_clusters)])
    return X, y, clusters
