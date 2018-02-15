#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
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

def conditional_sugiyama(x):
    p = .5 * (1 + math.tanh(x[0]+min(0, x[1])))
    y = np.random.choice([0, 1], p = [p, 1-p])
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
    
    y_train = conditional_expectation(conditional_sugiyama, X_train)
    y_test = conditional_expectation(conditional_sugiyama, X_test)

    return X_train, y_train, X_test, y_test

def in_hull(p, hull):
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def polygon_probability(X, n_clusters):
    ngrid = 1000
    rs = np.array(np.linspace(1e-10, 8, ngrid))
    hulls = []
    for n, r in np.ndenumerate(rs):
        V = vertices_polygon(n_clusters, r)
        hull = ConvexHull(V)
        hulls.append(hull)
    
    rds = []
    for c in range(n_clusters):
        X_c = X[c][0]
        nx = X_c.shape[0]
        rxs = np.zeros(shape = (nx, 1))
        for n in range(nx):
            rxmin = 100
            for h in range(ngrid):
                if in_hull(X_c[n], hulls[n]) == True:
                    rxmin = rs[h]
                    break
            rxs[n] = rxmin
        rds.append(rxs)
    return rds

def vertices_polygon(n_clusters, r):
    V = np.zeros(shape = (n_clusters, 2))
    for c in range(n_clusters):
        V[c,:] = [r * math.cos(2*math.pi*c/n_clusters), r * math.sin(2*math.pi*c/n_clusters)]
    return V

def get_population_parameters(n_clusters, r):
    mus = []
    covs = []
    V = vertices_polygon(n_clusters, r)
    for c in range(n_clusters):
        mu_0 = np.array([(6.0/10) * r * math.cos(2*math.pi*((c+.5)/n_clusters)), 
                    (6.0/10) * r * math.sin(2*math.pi*((c+.5)/n_clusters))])
        mu_1 = np.array([(12.0/10) * r * math.cos(2*math.pi*((c+.5)/n_clusters)), 
                    (12.0/10) * r * math.sin(2*math.pi*((c+.5)/n_clusters))])
        mus.append([mu_0, mu_1])
    
    for c in range(n_clusters):
        edge = V[int(math.fmod(c+1, n_clusters)),:]-V[c,:]
        edge = np.reshape(edge, (2, 1))
        mu_0 = mus[c][0]
        mu_0 = np.reshape(mu_0, (2,1))
        cov_0 = (1.0/16) * np.matmul(mu_0, mu_0.T) + (1.0/32) * np.matmul(edge, edge.T)
        cov_1 = (1.0/16) * np.matmul(mu_0, mu_0.T) + (1.0/32) * np.matmul(edge, edge.T)
        covs.append([cov_0, cov_1])
    return V, mus, covs

def get_data_experiment(n_clusters, n_samples, r):
    V, mus, covs = get_population_parameters(n_clusters, r)
    label_probs = np.array([.5, .5])

    X = []
    n_cluster = n_samples/n_clusters
    X_c = np.zeros(shape = (n_cluster, 3))
    for c in range(n_clusters):
        X_c = gaussian_mixture(n_cluster, np.array(mus[c]), 
            np.array([covs[c][0], covs[c][1]]), label_probs)
        X_cluster = np.repeat(c, n_cluster)
        X.append([X_c, X_cluster])
    return V, mus, covs, X

def plot_polygon(V, mus, X, n_clusters):
    hull = ConvexHull(V)
    for simplex in hull.simplices:
        plt.plot(V[simplex,0], V[simplex,1], 'k-')
    
    #plot data
    for n in range(n_clusters):
        X_n = X[n][0]
        plt.scatter(X_n[:,0], X_n[:,1])
    plt.axis('equal')
    plt.show()

def main():
    n_clusters, r = 7, 4
    n_samples = 500
    V, mus, _, X = get_data_experiment(n_clusters, n_samples, r)
    #plot_polygon(V, mus, X, n_clusters)

    #ver = polygon_probability(X, n_clusters)

if __name__ == '__main__':
    main()

    