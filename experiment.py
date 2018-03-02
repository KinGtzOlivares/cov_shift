#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import svm

import utils.get_data as data
import utils.plots as plots

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

def svm_classifier(config, data):
    X_train, y_train = data["X_train"], data["y_train"]
    C = 1.0  # SVM regularization parameter
    model = svm.SVC(kernel = 'linear',  gamma=0.7, C=C )
    model.fit(X_train, y_train)
    return model

def qda_propscores(n_clusters, df):
    # data
    X_propscore, y_propscore = df['X_propscore'], df['y_propscore']
    X_train, y_train = df['X_train'], df['y_train']
    # lda for the propscore training
    X_ps, y_ps, proxy_clusters = data.parse_data_propscore(n_clusters, X_propscore, y_propscore)
    model = QuadraticDiscriminantAnalysis(reg_param=0.0).fit(X_ps, proxy_clusters)
    # lda propscores for actual training
    X_train_ps, y_train_ps, _ = data.parse_data_propscore(n_clusters, X_train, y_train)
    # inference
    propscores = model.predict_proba(X_train_ps)
    return propscores

def discriminative_ratios(n_clusters, objective_c, propscores):
    objective = propscores[:, objective_c]
    complement = np.delete(propscores, objective_c, axis=1)
    complement = np.sum(complement, axis=1)
    objective_vall = np.column_stack((objective, complement))
    return objective_vall
    
def iw2s_logistic(n_clusters, data):
    X_propscore, X_train, X_test = data['X_propscore'], data['X_train'], data['X_test']
    lda_propscores(n_clusters, X_propscore, X_train)
    #return propscores

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
    
if __name__ == '__main__':
    main()