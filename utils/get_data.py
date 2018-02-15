#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.spatial #import ConvexHull
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

def get_population_parameters(n_sides, r):
    V_pyx = np.zeros(shape = (n_sides, 2))
    mus = [] #np.zeros(shape = (n_sides, 2, 2))
    covs = [] #np.zeros(shape = (n_sides, 2, 2, 2))
    for n in range(n_sides):
        V_pyx[n,:] = [r * math.cos(2*math.pi*n/n_sides), r * math.sin(2*math.pi*n/n_sides)]
        mu_0 = np.array([(6.0/10) * r * math.cos(2*math.pi*((n+.5)/n_sides)), 
                    (6.0/10) * r * math.sin(2*math.pi*((n+.5)/n_sides))])
        mu_1 = np.array([(12.0/10) * r * math.cos(2*math.pi*((n+.5)/n_sides)), 
                    (12.0/10) * r * math.sin(2*math.pi*((n+.5)/n_sides))])
        mus.append([mu_0, mu_1])
    
    for n in range(n_sides):
        edge = V_pyx[int(math.fmod(n+1,n_sides)),:]-V_pyx[n,:]
        edge = np.reshape(edge, (2, 1))
        mu_0 = mus[n][0]
        mu_0 = np.reshape(mu_0, (2,1))
        cov_0 = (1.0/16) * np.matmul(mu_0, mu_0.T) + (1.0/32) * np.matmul(edge, edge.T)
        cov_1 = (1.0/16) * np.matmul(mu_0, mu_0.T) + (1.0/32) * np.matmul(edge, edge.T)
        covs.append([cov_0, cov_1])
    return V_pyx, mus, covs

def get_data_experiment(n_clusters, n_samples, r):
    V_pyx, mus, covs = get_population_parameters(n_sides=n_clusters, r=r)
    label_probs = np.array([.5, .5])

    n_cluster = n_samples/n_clusters
    X_n = np.zeros(shape = (n_cluster, 3))
    X = []
    for n in range(n_clusters):
        #print(type(covs[n])); print(len(covs[n])); print(covs[n][0].shape)
        X_n = gaussian_mixture(n_cluster, np.array(mus[n]), np.array([covs[n][0], covs[n][1]]), label_probs)
        X_cluster = np.repeat(n, n_cluster)
        X.append([X_n, X_cluster])
    return V_pyx, mus, covs, X

def plot_polygon(V, mus, X, n_clusters):
    #plot convex hull
    hull = scipy.spatial.ConvexHull(V)
    for simplex in hull.simplices:
        plt.plot(V[simplex,0], V[simplex,1], 'k-')
    
    #plot data
    for n in range(n_clusters):
        mu_0, mu_1 = mus[n][0], mus[n][1]
        plt.plot(mu_0[0], mu_0[1], 'o')
        plt.plot(mu_1[0], mu_1[1], 's')
    for n in range(n_clusters):
        X_n = X[n][0]
        plt.scatter(X_n[:,0], X_n[:,1])
    plt.axis('equal')
    plt.show()

def main():
    #import utils.get_data as data
    #import utils.plots as plots
    n_clusters, r = 7, 4
    n_samples = 500
    V_pyx, mus, _, X = get_data_experiment(n_clusters, n_samples, r)
    plot_polygon(V_pyx, mus, X, n_clusters)

if __name__ == '__main__':
    main()

    