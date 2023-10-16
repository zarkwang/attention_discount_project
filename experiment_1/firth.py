#!/usr/bin/env python
'''Firth's bias-reduced logistic regression 
This code is modified from the work of John Lees
See https://www.ncbi.nlm.nih.gov/pubmed/12758140'''

import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# loss function
def firth_likelihood(beta, logit):
    return -(logit.loglike(beta) + 0.5*np.log(np.linalg.det(-logit.hessian(beta))))

# Fit model via firth regression
# Note information = -hessian, for some reason available but not implemented in statsmodels
def fit_firth(y, X, 
              start_vec=None, step_limit=1000, convergence_limit=1e-5):

    logit_model = sm.Logit(y,X)
    
    if start_vec is None:
        start_vec = np.zeros(X.shape[1])
    
    beta_iterations = []
    beta_iterations.append(start_vec)
    for i in range(0, step_limit):
        pi = logit_model.predict(beta_iterations[i])
        
        # inverse of the Fisher information matrix
        var_covar_mat = np.linalg.pinv(-logit_model.hessian(beta_iterations[i]))

        # build hat matrix
        W = np.diagflat(np.multiply(pi, 1-pi))
        rootW = np.sqrt(W)
        H = np.dot(np.transpose(X), np.transpose(rootW))
        H = np.matmul(var_covar_mat, H)
        H = np.matmul(np.dot(rootW, X), H)

        # penalised score
        U = np.matmul(np.transpose(X), y - pi + np.multiply(np.diagonal(H), 0.5 - pi))

        # parameter updating
        new_beta = beta_iterations[i] + np.matmul(var_covar_mat, U)

        # step halving
        j = 0
        while firth_likelihood(new_beta, logit_model) > firth_likelihood(beta_iterations[i], logit_model):
            new_beta = beta_iterations[i] + 0.5*(new_beta - beta_iterations[i])
            j = j + 1
            if (j > step_limit):
                sys.stderr.write('Firth regression failed\n')
                return None

        beta_iterations.append(new_beta)

        print('iteration:',i,', LL=',firth_likelihood(beta_iterations[i], logit_model))

        if i > 0 and (np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) < convergence_limit):
            break

    return_fit = None
    if np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) >= convergence_limit:
        sys.stderr.write('Firth regression failed\n')
    else:
        # Calculate stats
        fitll = -firth_likelihood(beta_iterations[-1], logit_model)
        intercept = beta_iterations[-1][0]
        beta = beta_iterations[-1][1:].tolist()
        bse = np.sqrt(np.diagonal(np.linalg.pinv(-logit_model.hessian(beta_iterations[-1]))))
        
        return_fit = intercept, beta, bse, fitll

    return return_fit



class firthLogit:

    def __init__(self,y,X):
        self.y = y
        self.X = X
        # Note a constant variable (=1) needs to be added to X
    
    def fit(self):
        (intercept, beta, bse, fitll) = fit_firth(self.y,self.X,
                                                start_vec = None, 
                                                step_limit = 1000, 
                                                convergence_limit = 1e-5)
        
        self.coef = [intercept] + beta
        self.bse = bse
        self.fitll = fitll

    # Wald test
    def wald(self, confidence_level=0.95):

        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha / 2)

        waldp = []
        wald_lower = []
        wald_upper = []
        for coef_val, bse_val in zip(self.coef, self.bse):
            waldp.append(2 * (1 - stats.norm.cdf(abs(coef_val/bse_val))))
            wald_lower.append(coef_val - z_critical * bse_val)
            wald_upper.append(coef_val + z_critical * bse_val)

        wald_result = {'var_name':self.X.columns,
                       'coef':self.coef,
                       'bse':self.bse,
                       'wald_p':waldp,
                       'lower_bound':wald_lower,
                       'upper_bound':wald_upper}
        
        print('Confidence level: ',confidence_level)

        return pd.DataFrame(wald_result)

    # Likelihood Ratio Test: p-value
    def LRT(self):
        lrtp = []
        for beta_idx, (coef_val, bse_val) in enumerate(zip(self.coef, self.bse)):
            null_X = np.delete(self.X, beta_idx, axis=1)
            (null_intercept, null_beta, null_bse, null_fitll) = fit_firth(self.y, null_X)
            lrstat = -2*(null_fitll - self.fitll)
            lrt_pvalue = 1
            if lrstat > 0: # non-convergence
                lrt_pvalue = stats.chi2.sf(lrstat, 1)
            lrtp.append(lrt_pvalue)
        
        return lrtp
    
    # Prediction
    def predict(self,method = None):

        if method is None:
            linear_sum = self.coef @ self.X.T
        
        elif method == 'flic': # using the FLIC method to reduce prediction bias
            
            # calculate b1 * X1 + ... + bm * Xm (removing intercept)
            linear_sum_no_intercept = self.coef[1:] @ self.X.iloc[:,1:].T
        
            # logistic regression
            X_ = sm.add_constant(linear_sum_no_intercept)
            logit_flic = sm.Logit(self.y,X_).fit()

            print('original intercept: ',self.coef[0])
            print('new intercept: ',logit_flic.params[0])
        
            linear_sum = logit_flic.params[0] + linear_sum_no_intercept
        
        pred = 1/(1+np.exp(-linear_sum))

        return pred
