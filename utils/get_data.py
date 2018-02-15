#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
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
    rs = np.array(np.linspace(1e-10, 8, n_grid))
    Vs = []
    for n, r in np.ndenumerate(rs):
        V = vertices_polygon(n_clusters, r)
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
            for h in range(n_grid):
                if in_hull(Vs[n], X_c[n]) == True:
                    rxsup = rs[h]
                    break
            rxs[n] = rxsup
            p = .5 * (1 + math.tanh(rxsup-r))
            ys[n] = np.random.choice([0, 1], p = [p, 1-p])
        rds.append(rxs)
        Y.append(ys)
    return Y, rds

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

def get_data_experiment(n_samples, n_clusters, r, n_grid):
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
    
    Y, _ = conditional_polygon(X, n_clusters, r, n_grid)

    return V, mus, covs, X, Y

def plot_polygon(V, mus, X, Y, n_clusters):
    hull = ConvexHull(V)
    for simplex in hull.simplices:
        plt.plot(V[simplex,0], V[simplex,1], 'k-')
    
    #plot data
    for c in range(n_clusters):
        X_c, y_c = X[c][0], Y[c]
        y_c = y_c.flatten().tolist()
        colors_c = ["#E99F51" if y <=0 else "#386D9D" for y in y_c]
        plt.scatter(X_c[:,0], X_c[:,1], c=colors_c)
    plt.axis('equal')
    plt.show()

def main():
    n_clusters, r, n_grid = 6, 4, 20
    n_samples = 70
    V, mus, covs, X, Y = get_data_experiment(n_samples, n_clusters, r, n_grid)
    plot_polygon(V, mus, X, Y, n_clusters)

    #n_grid = 20
    #ver = conditional_polygon(X, n_clusters, n_grid)
    #print(len(ver))
    #print(ver[0])

if __name__ == '__main__':
    main()

    