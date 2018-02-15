#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import svm

import utils.get_data as data
import utils.plots as plots

#mu_vec1 = np.array([0,0])
#cov_mat1 = np.array([[2,0],[0,2]])
#x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 100)
#mu_vec1 = mu_vec1.reshape(1,2).T # to 1-col vector
#
#mu_vec2 = np.array([1,2])
#cov_mat2 = np.array([[1,0],[0,1]])
#x2_samples = np.random.multivariate_normal(mu_vec2, cov_mat2, 100)
#mu_vec2 = mu_vec2.reshape(1,2).T
#
#
#fig = plt.figure()
#
#
#plt.scatter(x1_samples[:,0],x1_samples[:,1], marker='+')
#plt.scatter(x2_samples[:,0],x2_samples[:,1], c= 'green', marker='o')
#
#X = np.concatenate((x1_samples,x2_samples), axis = 0)
#Y = np.array([0]*100 + [1]*100)
#
def svm_classifier(config, data):
    X_train, y_train = data["X_train"], data["y_train"]
    C = 1.0  # SVM regularization parameter
    model = svm.SVC(kernel = 'linear',  gamma=0.7, C=C )
    model.fit(X_train, y_train)
    #
    #w = clf.coef_[0]
    #a = -w[0] / w[1]
    #xx = np.linspace(-5, 5)
    #yy = a * xx - (clf.intercept_[0]) / w[1]
    #
    #plt.plot(xx, yy, 'k-')
    #plt.show()
    return model


def main():
    import utils.get_data as data
    import utils.plots as plots

    np.random.seed(seed=1234)
    root = '/Users/kdgutier/Desktop/cov_shift/'
    
    n_train, n_test = 60, 30
    X_train, y_train, X_test, y_test = data.get_data_sugiyama(n_train, n_test)

    data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
    config = {"n_train": n_train, "n_test": n_test}
    

    model = svm_classifier(config, data)
    plots.graph_data_sugiyama1(config, data, path = root+'images/sugiyama1.png')
    plots.graph_data_sugiyama2(config, data, model0 = model, path = root+'images/sugiyama2.png')


if __name__ == '__main__':
    main()