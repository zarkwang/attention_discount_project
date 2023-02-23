
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize
from mpl_utility_func import mpl_v


def choice_prob(ss_t, ss_x, ll_t, ll_x, params, dstyle, ustyle):

    ss_v = mpl_v(ss_t,ss_x,params,dstyle,ustyle)
    ll_v = mpl_v(ll_t,ll_x,params,dstyle,ustyle)
    p_choice_ll = 1/(1+np.exp(ss_v - ll_v))
    
    return p_choice_ll


def log_likelihood(params, data, dstyle, ustyle):

    ss_t = data['ss_delay'].values
    ss_x = data['ss_amount'].values
    ll_t = data['ll_delay'].values
    ll_x = data['ll_amount'].values
    choice = data['choice'].values
    
    p_choice_ll = choice_prob(ss_t, ss_x, ll_t, ll_x, params, dstyle, ustyle)
    p_choice_ss = 1 - p_choice_ll
    p_choice = np.where(choice == 1, p_choice_ll, p_choice_ss)

    log_like = np.sum(np.log(p_choice))
    return -log_like


# estimation with maximum likelihood method
def mle(data,init_params,dstyle,ustyle,bounds):
    
    result = minimize(log_likelihood, x0=init_params, args=(data,dstyle,ustyle), bounds=bounds,
                        method='L-BFGS-B')
    
    if result.success:
        se = np.sqrt(np.diag(result.hess_inv.todense())) / np.sqrt(len(data))
        log_like = -result.fun
        aic = 2*len(init_params)-2*log_like
        bic = 2*np.log(len(data))*len(init_params)-2*log_like
        gradient = result.jac
    else:
        print("Fail to converge.")
        
    return {'params':result.x,'se':se,'log-likelihood':log_like,'aic':aic,'bic':bic,'gradient':gradient}


# estimation with Bayesian method

