import sys
sys.path.append("/home/ec2-user/experiment/AIF360/")
from scipy.optimize import minimize # for loss func minimization
import numpy as np
import aif360.algorithms.inprocessing.zvrg.zvrg_utils as zvrg_ut
import aif360.algorithms.inprocessing.zvrg.zvrg_loss_funcs as zvrg_lf
import aif360.algorithms.inprocessing.gyf.gyf_utils as gyf_ut
import aif360.algorithms.inprocessing.gyf.gyf_loss_funcs as gyf_lf

import random
random.seed()
np.seterr(divide = 'ignore', invalid='ignore')

####################################################
# tools
####################################################
# Sigmoid function
def sigmoid(inx):
    if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))

# log loss
def log_logistic(X):

	""" This function is used from scikit-learn source code. Source link below """

	"""Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
	This implementation is numerically stable because it splits positive and
	negative values::
	    -log(1 + exp(-x_i))     if x_i > 0
	    x_i - log(1 + exp(x_i)) if x_i <= 0

	Parameters
	----------
	X: array-like, shape (M, N)
	    Argument to the logistic function

	Returns
	-------
	out: array, shape (M, N)
	    Log of the logistic function evaluated at every point in x
	Notes
	-----
	Source code at:
	https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
	-----

	See the blog post describing this implementation:
	http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
	"""
	if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
	out = np.empty_like(X) # same dimensions and data types

	idx = X>0
	out[idx] = -np.log(1.0 + np.exp(-X[idx]))
	out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
	return out

####################################################
# algorithms
####################################################
# ZVRG
def zvrg(features, labels, index, C, thresh):
    N = features.shape[0]
    d = features.shape[1]
    # X = np.asarray(features)
    X = np.zeros([N, d+1])
    X[:,0:d] = features
    X[:,d] = [1.0 for i in range(N)]
    X = np.delete(X, index, 1)

    # loss_function = lf._rosen
    loss_function = zvrg_lf._rosen
    x_control_train = {"attr": features[:,index]}
    mode = {"fairness": 1, "lam": float(C), 'is_reg': 1}
    sensitive_attrs = list(x_control_train.keys())
    sensitive_attrs_to_cov_thresh = dict((k, thresh) for (k, v) in x_control_train.items())
    theta = zvrg_ut.train_model(X, labels, x_control_train, loss_function,
                           mode.get('fairness', 0),
                           mode.get('accuracy', 0),
                           mode.get('separation', 0),
                           sensitive_attrs,
                           sensitive_attrs_to_cov_thresh,
                           mode.get('gamma', None),
                           mode.get('lam', None),
                           mode.get('is_reg', 0))
    return theta


#####################################################
def gyf(features, labels, index, C, thresh):
    N = features.shape[0]
    d = features.shape[1]
    # X = np.asarray(features)
    X = np.zeros([N, d + 1])
    X[:, 0:d] = features
    X[:, d] = [1.0 for i in range(N)]
    X = np.delete(X, index, 1)

    # loss_function = lf._logistic_loss_l2_reg
    loss_function = gyf_lf._fair_logistic_loss_l2
    x_control_train = {"attr": features[:, index]}
    mode = {"fairness": 2, "lam": float(C), 'is_reg': 1, 'gamma': float(thresh)}
    sensitive_attrs = list(x_control_train.keys())
    sensitive_attrs_to_cov_thresh = {}
    theta = gyf_ut.train_model(X, labels, x_control_train, loss_function,
                                mode.get('fairness', 0),
                                mode.get('accuracy', 0),
                                mode.get('separation', 0),
                                sensitive_attrs,
                                sensitive_attrs_to_cov_thresh,
                                mode.get('gamma', 1),
                                mode.get('lam', None),
                                mode.get('is_reg', 0))
    return theta

###############################################################
def undenoised(features, labels, index, C, tau):
    N = features.shape[0]
    d = features.shape[1]
    N1 = sum(features[:, index])
    N0 = N - N1
    X = np.zeros([N, d + 1])
    X[:, 0:d] = features
    X[:, d] = [1.0 for i in range(N)]
    X = np.delete(X, index, 1)

    # # logistic loss
    # def logistic_loss_l2_reg(w):
    #     yz = np.zeros(N)
    #     for i in range(N):
    #         yz[i] = (2 * labels[i] - 1) * np.dot(X[i], w)
    #     # Logistic loss is the negative of the log of the logistic function.
    #     logistic_loss = -np.sum(log_logistic(yz))
    #     l2_reg = (float(C) * N) * np.sum([elem * elem for elem in w])
    #     out = logistic_loss + l2_reg
    #     return out

    # loss function
    def rosen(x):
        obj = 0
        indices = list(range(N))
        for i in indices:
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            obj -= label * np.log(sigma) + (1-label) * np.log(1-sigma)
        obj /= len(indices)
        for i in range(d):
            obj += C * x[i]**2
        return obj

    def rosen_der(x):
        der = np.zeros(d)
        indices = list(range(N))
        for i in indices:
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            der += (sigma-label) * fea
        der = der / len(indices)
        for i in range(d):
            der[i] += 2 * C * x[i]
        return der

    # fairness constraints
    def cons_f(x):
        f = np.zeros(2)
        product = np.array([np.dot(x, X[i]) for i in range(N)])
        symbol = np.zeros(N)
        for i in range(N):
            if product[i] >= 0:
                symbol[i] = 1
            else:
                symbol[i] = 0
        for i in range(N):
            if int(features[i][index]) == 0:
                f[0] += symbol[i] / N0
                f[1] += - tau * symbol[i] / N0
            else:
                f[0] += - tau * symbol[i] / N1
                f[1] += symbol[i] / N1
        return f

    # testing
    def testing(theta):
        NTrue = 0
        for i in range(N):
            predict = np.dot(theta, X[i])
            # seed = random.random()
            # if seed < predict:
            if predict >= 0:
                predict = 1
            else:
                predict = 0
            if labels[i] == predict:
                NTrue += 1
        acc = NTrue / N
        return acc

    lower = max(np.sum(labels) / N, (N - np.sum(labels)) / N)
    flag = 0

    while flag == 0:
        x0 = np.random.rand(d)
        ineq_cons = {'type': 'ineq', 'fun': lambda x: cons_f(x)}
        res = minimize(fun=rosen, x0=x0, method='SLSQP', jac=rosen_der, constraints=[ineq_cons],
                       options={'maxiter': 500, 'ftol': 1e-1, 'eps': 1e-2, 'disp': True})
        acc = testing(res.x)
        if acc >= lower - 0.01:
            flag = 1

    return res.x

###############################################################
def undenoised_eo(features, labels, index, C, tau):
    N = features.shape[0]
    Ny = len(labels) - sum(labels) # number of Y=0
    d = features.shape[1]
    N1 = sum([1 if features[:,index][i] == 1 and labels[i] == 0 else 0 for i in range(len(labels))]) # Number of Y=0, hat{Z}=1
    N0 = Ny - N1  # Number of Y=0, hat{Z}=0
    X = np.zeros([N, d + 1])
    X[:, 0:d] = features
    X[:, d] = [1.0 for i in range(N)]
    X = np.delete(X, index, 1)

    # # logistic loss
    # def logistic_loss_l2_reg(w):
    #     yz = np.zeros(N)
    #     for i in range(N):
    #         yz[i] = (2 * labels[i] - 1) * np.dot(X[i], w)
    #     # Logistic loss is the negative of the log of the logistic function.
    #     logistic_loss = -np.sum(log_logistic(yz))
    #     l2_reg = (float(C) * N) * np.sum([elem * elem for elem in w])
    #     out = logistic_loss + l2_reg
    #     return out

    # loss function
    def rosen(x):
        obj = 0
        indices = list(range(N))
        for i in indices:
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            obj -= label * np.log(sigma) + (1-label) * np.log(1-sigma)
        obj /= len(indices)
        for i in range(d):
            obj += C * x[i]**2
        return obj

    def rosen_der(x):
        der = np.zeros(d)
        indices = list(range(N))
        for i in indices:
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            der += (sigma-label) * fea
        der = der / len(indices)
        for i in range(d):
            der[i] += 2 * C * x[i]
        return der

    # fairness constraints
    def cons_f(x):
        f = np.zeros(2)
        product = np.array([np.dot(x, X[i]) for i in range(N)])
        symbol = np.zeros(N)
        for i in range(N):
            if product[i] >= 0:
                symbol[i] = 1
            else:
                symbol[i] = 0
        for i in range(N):
            if labels[i] == 1:
                continue
            if int(features[i][index]) == 0:
                f[0] += symbol[i] / N0
                f[1] += - tau * symbol[i] / N0
            else:
                f[0] += - tau * symbol[i] / N1
                f[1] += symbol[i] / N1
        return f

    # testing
    def testing(theta):
        NTrue = 0
        for i in range(N):
            predict = np.dot(theta, X[i])
            # seed = random.random()
            # if seed < predict:
            if predict >= 0:
                predict = 1
            else:
                predict = 0
            if labels[i] == predict:
                NTrue += 1
        acc = NTrue / N
        return acc

    lower = max(np.sum(labels) / N, (N - np.sum(labels)) / N)
    flag = 0

    while flag == 0:
        x0 = np.random.rand(d)
        ineq_cons = {'type': 'ineq', 'fun': lambda x: cons_f(x)}
        res = minimize(fun=rosen, x0=x0, method='SLSQP', jac=rosen_der, constraints=[ineq_cons],
                       options={'maxiter': 500, 'ftol': 1e-1, 'eps': 1e-2, 'disp': True})
        acc = testing(res.x)
        if acc >= lower - 0.01:
            flag = 1

    return res.x

#####################################################
# solving denoised fair program
def denoised(features, labels, index, C, tau, lam, eta0, eta1):
    N = features.shape[0]
    d = features.shape[1]
    N1 = sum(features[:,index])
    N0 = N - N1
    mu0 = (1 - eta1) * N0 / N - eta1 * N1 / N
    mu1 = (1 - eta0) * N1 / N - eta0 * N0 / N
    coeff00 = (tau - 0.01) * eta0 * mu0 + (1 - eta1) * mu1
    coeff01 = - (tau - 0.01) * (1 - eta0) * mu0 - eta1 * mu1
    coeff10 = - (tau - 0.01) * (1 - eta1) * mu1 - eta0 * mu0
    coeff11 = (tau - 0.01) * eta1 * mu1 + (1 - eta0) * mu0
    per = (1 - eta0 - eta1) * lam - 0.01
    X = np.zeros([N, d+1])
    X[:,0:d] = features
    X[:,d] = [1.0 for i in range(N)]
    X = np.delete(X, index, 1)

    # loss function
    def rosen(x):
        obj = 0
        indices = list(range(N))
        for i in indices:
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            obj -= label * np.log(sigma) + (1-label) * np.log(1-sigma)
        obj /= len(indices)
        for i in range(d):
            obj += C * x[i]**2
        return obj

    def rosen_der(x):
        der = np.zeros(d)
        indices = list(range(N))
        for i in indices:
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            der += (sigma-label) * fea
        der = der / len(indices)
        for i in range(d):
            der[i] += 2 * C * x[i]
        return der

    # denoised fairness constraints
    def cons_f(x):
        f = np.zeros(4)
        product = np.array([np.dot(x, X[i]) for i in range(N)])
        symbol = np.zeros(N)
        for i in range(N):
            if product[i] >= 0:
                symbol[i] = 1
            else:
                symbol[i] = 0
        for i in range(N):
            if int(features[i][index]) == 0:
                f[0] += (1-eta1) * symbol[i] / N
                f[1] += - eta0 * symbol[i] / N
                f[2] += coeff00 * symbol[i]
                f[3] += coeff10 * symbol[i]
            else:
                f[0] += - eta1 * symbol[i] / N
                f[1] += (1 - eta0) * symbol[i] / N
                f[2] += coeff01 * symbol[i]
                f[3] += coeff11 * symbol[i]
        f[0] -= per
        f[1] -= per
        return f

    # testing
    def testing(theta):
        NTrue = 0
        for i in range(N):
            predict = np.dot(theta, X[i])
            # seed = random.random()
            # if seed < predict:
            if predict >= 0:
                predict = 1
            else:
                predict = 0
            if labels[i] == predict:
                NTrue += 1
        acc = NTrue / N
        return acc

    lower = max(np.sum(labels) / N, (N - np.sum(labels)) / N)
    flag = 0

    c = 0
    print ("Started")
    while flag == 0:
        x0 = np.random.rand(d)
        ineq_cons = {'type': 'ineq', 'fun' : lambda x: cons_f(x)}
        # res = minimize(logistic, x0, method='SLSQP', jac=logistic_der, constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': True})
        res = minimize(fun = rosen, x0 = x0, method='SLSQP', jac = rosen_der, constraints = [ineq_cons],
                       options = {'maxiter': 500, 'ftol': 5e-3, 'eps' : 1e-2, 'disp': True})
        acc = testing(res.x)
        if acc >= lower - 0.01:
            flag = 1

        print (c, acc)
        c += 1
        if acc > 0.74:
            flag =1
    return res.x

#####################################################
# solving denoised fair program
def denoised_eo(features, labels, index, C, tau, lam, eta0, eta1):
    N = features.shape[0]    
    Ny = len(labels) - sum(labels) # number of Y=0
    d = features.shape[1]
    N1 = sum([1 if features[:,index][i] == 1 and labels[i] == 0 else 0 for i in range(len(labels))]) # Number of Y=0, hat{Z}=1
    N0 = Ny - N1  # Number of Y=0, hat{Z}=0
    mu0 = (1 - eta1) * N0 / Ny - eta1 * N1 / Ny # Pr[Y=0, \hat{Z} = 0]
    mu1 = (1 - eta0) * N1 / Ny - eta0 * N0 / Ny # Pr[Y=0, \hat{Z} = 1]
    coeff00 = (tau - 0.01) * eta0 * mu0 + (1 - eta1) * mu1
    coeff01 = - (tau - 0.01) * (1 - eta0) * mu0 - eta1 * mu1
    coeff10 = - (tau - 0.01) * (1 - eta1) * mu1 - eta0 * mu0
    coeff11 = (tau - 0.01) * eta1 * mu1 + (1 - eta0) * mu0
    per = (1 - eta0 - eta1) * lam - 0.01
    X = np.zeros([N, d+1])
    X[:,0:d] = features
    X[:,d] = [1.0 for i in range(N)]
    X = np.delete(X, index, 1)

    # loss function
    def rosen(x):
        obj = 0
        indices = list(range(N))
        for i in indices:
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            obj -= label * np.log(sigma) + (1-label) * np.log(1-sigma)
        obj /= len(indices)
        for i in range(d):
            obj += C * x[i]**2
        return obj

    def rosen_der(x):
        der = np.zeros(d)
        indices = list(range(N))
        for i in indices:
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            der += (sigma-label) * fea
        der = der / len(indices)
        for i in range(d):
            der[i] += 2 * C * x[i]
        return der

    # denoised fairness constraints
    def cons_f(x):
        f = np.zeros(4)
        product = np.array([np.dot(x, X[i]) for i in range(N)])
        symbol = np.zeros(N)
        for i in range(N):
            if product[i] >= 0:
                symbol[i] = 1
            else:
                symbol[i] = 0
        for i in range(N):
            if labels[i] == 1:
                continue
            if int(features[i][index]) == 0:
                f[0] += (1-eta1) * symbol[i] / Ny
                f[1] += - eta0 * symbol[i] / Ny
                f[2] += coeff00 * symbol[i]
                f[3] += coeff10 * symbol[i]
            else:
                f[0] += - eta1 * symbol[i] / Ny
                f[1] += (1 - eta0) * symbol[i] / Ny
                f[2] += coeff01 * symbol[i]
                f[3] += coeff11 * symbol[i]
        f[0] -= per
        f[1] -= per
        return f

    # testing
    def testing(theta):
        NTrue = 0
        for i in range(N):
            predict = np.dot(theta, X[i])
            # seed = random.random()
            # if seed < predict:
            if predict >= 0:
                predict = 1
            else:
                predict = 0
            if labels[i] == predict:
                NTrue += 1
        acc = NTrue / N
        return acc

    lower = max(np.sum(labels) / N, (N - np.sum(labels)) / N)
    flag = 0

    c = 0
    print ("Started")
    while flag == 0:
        x0 = np.random.rand(d)
        ineq_cons = {'type': 'ineq', 'fun' : lambda x: cons_f(x)}
        # res = minimize(logistic, x0, method='SLSQP', jac=logistic_der, constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': True})
        res = minimize(fun = rosen, x0 = x0, method='SLSQP', jac = rosen_der, constraints = [ineq_cons],
                       options = {'maxiter': 500, 'ftol': 5e-3, 'eps' : 1e-1, 'disp': True})
        acc = testing(res.x)
        if acc >= lower - 0.01:
            flag = 1

        print (c, acc)
        c += 1
        
        if c >= 3:
            flag =1
    return res.x





# test