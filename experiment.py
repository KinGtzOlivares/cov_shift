#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import svm

import utils.get_data as data
import utils.plots as plots

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import linear_model

def objective_vall(n_clusters, objective_c, propscores):
    proxy_clusters = [objective_c, objective_c + n_clusters]
    objective = propscores[:, proxy_clusters]
    objective = np.sum(objective, axis=1)
    complement = np.delete(propscores, proxy_clusters, axis=1)
    complement = np.sum(complement, axis=1)
    objective_vall = np.column_stack((objective, complement))
    return objective_vall

def qda_propscores(n_clusters, df, is_test=False):
    # data
    X_propscore, y_propscore = df['X_propscore'], df['y_propscore']
    X_train, y_train = df['X_train'], df['y_train']
    X_test, y_test = df['X_test'], df['y_test']
    # lda for the propscore training
    X_ps, y_ps, proxy_clusters = data.parse_data_propscore(n_clusters, X_propscore, y_propscore)
    model = QuadraticDiscriminantAnalysis(reg_param=0.0).fit(X_ps, proxy_clusters)
    # lda propscores for actual training

    # inference
    if is_test==True:
        X_test_ps, _ , _ = data.parse_data_propscore(n_clusters, X_test, y_test)
        propscores = model.predict_proba(X_test_ps)
    else:
        X_train_ps, _ , _ = data.parse_data_propscore(n_clusters, X_train, y_train)
        propscores = model.predict_proba(X_train_ps)
    
    return propscores

def correct_propscores(n_clusters, propscores):
    correct_propscores = np.zeros(shape=(propscores.shape[0], n_clusters))
    for c in range(n_clusters):
        correct_propscores[:, c] = propscores[:, c] + propscores[:, (c + n_clusters)]
    return correct_propscores

def naive_logistic(n_clusters, df):
    X_train, y_train, _ = data.parse_data(n_clusters, df['X_train'], df['y_train'])
    X_test, y_test, _ = data.parse_data(n_clusters, df['X_test'], df['y_test'])
    # model
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    # inference
    y_hat = model.predict(X_test)
    return y_hat

def weighted_logistic(n_clusters, df, weights):
    X_train, y_train, _ = data.parse_data(n_clusters, df['X_train'], df['y_train'])
    X_test, _ , _ = data.parse_data(n_clusters, df['X_test'], df['y_test'])
    # model
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train, sample_weight=weights)
    # inference
    y_hat = model.predict(X_test)
    return y_hat

def pure_logistic(n_clusters, df):
    X_train, y_train, clusters_train = data.parse_data(n_clusters, df['X_train'], df['y_train'])
    X_test, y_test, clusters_test = data.parse_data(n_clusters, df['X_test'], df['y_test'])
    
    y_hat = np.zeros(shape=(X_test.shape[0],))
    for c in range(n_clusters):
        X_train_c, y_train_c = X_train[clusters_train == c], y_train[clusters_train == c]
        X_test_c = X_test[clusters_test == c]
        model = linear_model.LogisticRegression()
        model.fit(X_train_c, y_train_c)
        y_hat_c = model.predict(X_test_c)
        y_hat[clusters_test == c] = y_hat_c
    return y_hat

def dm_logistic(n_clusters, df):
    X_train, y_train, _ = data.parse_data(n_clusters, df['X_train'], df['y_train'])
    X_test, _ , _ = data.parse_data(n_clusters, df['X_test'], df['y_test'])
    
    # Train
    propscores = qda_propscores(n_clusters, df)
    y_hat_models = np.zeros(shape=(X_test.shape[0], n_clusters))
    y_hat = np.zeros(shape=X_test.shape[0])
    for c in range(n_clusters):
        prop_weights = objective_vall(n_clusters, c, propscores)[:,0]
        model = linear_model.LogisticRegression()
        model.fit(X_train, y_train, sample_weight=prop_weights)
        y_hat_c = model.predict(X_test)
        y_hat_models[:,c] = y_hat_c
    
    # Test
    propscores = qda_propscores(n_clusters, df, is_test=True)
    propscores = correct_propscores(n_clusters, propscores)
    model_selector = np.argmax(propscores, axis=1)
    for idx in range(X_test.shape[0]):
        y_hat[idx] = y_hat_models[idx, model_selector[idx]-1]
    return y_hat

def svm_benchmark(df):
    pass

def goodness_fit(y_hat, y):
    n_test = y.shape[0]
    true_p = np.sum(np.multiply(1*(y_hat==1), 1*(y==1)))
    false_p = np.sum(np.multiply(y_hat==1,y==0))
    true_n = np.sum(np.multiply(y_hat==0,y==0))
    false_n = np.sum(np.multiply(y_hat==0,y==1))
    
    assert (true_p + false_p + true_n + false_n) == n_test
    precision = true_p // (true_p+false_p)
    recall = true_p // (true_p+false_n)
    accuracy = (true_p + true_n) // n_test
    
    print(precision); print(recall); print(accuracy)
    return precision, recall, accuracy

def compare_models(n_clusters, df):
    _, y_test, _ = data.parse_data(n_clusters, df['X_test'], df['y_test'])
    
    y_hat_naive = naive_logistic(n_clusters, df)
    y_hat_pure = pure_logistic(n_clusters, df)
    y_hat_dm = dm_logistic(n_clusters, df)
    
    #table = np.zeros(shape=(3,3))
    t1 = goodness_fit(y_hat_naive, y_test)
    t2 = goodness_fit(y_hat_naive, y_test)
    t3 = goodness_fit(y_hat_naive, y_test)
    table = np.vstack((t1,t2,t3))
    return table

def main():
    import utils.get_data as data
    import utils.plots as plots

    #
    # n_train, n_test = 60, 30
    # X_train, y_train, X_test, y_test = data.get_data_sugiyama(n_train, n_test)
    #
    # data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
    # config = {"n_train": n_train, "n_test": n_test}
    #
    #
    # model = svm_classifier(config, data)
    # plots.graph_data_sugiyama1(config, data, path = root+'images/sugiyama1.png')
    # plots.graph_data_sugiyama2(config, data, model0 = model, path = root+'images/sugiyama2.png')

    # np.random.seed(seed=1234)
    # root = '/Users/kdgutier/Desktop/cov_shift/'

    # n_clusters, r, n_grid = 6, 4, 40 #20
    # n_samples = 1200 #120 #3600
    # X, Y, V = data.get_data_experiment(n_samples, n_clusters, r, n_grid)
    # data.save_data_experiment(n_clusters, X, Y, V, path=(root + 'data/'))
    
    np.random.seed(seed=1234)
    root = '/Users/kdgutier/Desktop/cov_shift/'

    n_clusters = 6
    X, Y, V = data.load_data_experiment(n_clusters, path=(root + 'data/'))
    plots.graph_data_experiment(n_clusters, X, Y, V, path=(root+'images/experiment2.png') )

    datos = data.split_data_experiment(n_clusters, X, Y)
    propscores = qda_propscores(n_clusters, datos)
    plots.graph_first_stage(n_clusters, df=datos, V=V, propscores=propscores, path=(root+'images/experiment_1stage.png'))
    ver = compare_models(n_clusters, datos)
    print(ver)
    
if __name__ == '__main__':
    main()