import yaml
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize,basinhopping
from mpl import choice_rule

with open('config_param.yaml', 'r') as f:
        config_param = yaml.safe_load(f)


def constraint(params,bound):

    cons = True

    for i in range(len(params)):
     cons = (params[i] <= bound[i][1]) & (params[i] >= bound[i][0]) & cons

    return cons



def log_likelihood(params, data, dstyle, ustyle, bounds, method="logit", regenerate_sample=True,
                   kwargs={}):

    if not constraint(params,bounds):
        return 1e10 
    else:
        temper = params[-1]
        ss_t = data['ss_delay'].values
        ss_x = data['ss_amount'].values
        ll_t = data['ll_delay'].values
        ll_x = data['ll_amount'].values
        choice = data['choice'].values
        
        if not regenerate_sample:
            p_choice_ll = choice_rule.choice_prob_du(ss_t, ss_x, ll_t, ll_x, params, dstyle, ustyle, temper, method,
                                                     regenerate_sample=False, simu_sample = kwargs['simu_sample'])
        else:
            p_choice_ll = choice_rule.choice_prob_du(ss_t, ss_x, ll_t, ll_x, params, dstyle, ustyle, temper, method)
        
        p_choice_ll[np.where(p_choice_ll == 1)] = p_choice_ll[np.where(p_choice_ll == 1)] - 1e-8
        p_choice_ll[np.where(p_choice_ll == 0)] = p_choice_ll[np.where(p_choice_ll == 0)] + 1e-8
        
        p_choice_ss = 1 - p_choice_ll
        p_choice = np.where(choice == 1, p_choice_ll, p_choice_ss)

        log_like = np.sum(np.log(p_choice))

        return -log_like





def mle(style,data,disp_output=False,disp_step=False,simu_size=1000):

    dstyle = style['dstyle']
    ustyle = style['ustyle']
    method = style['method']

    x0 = config_param['discount_func'][dstyle]["x0"] + \
         config_param['utility_func'][ustyle]["x0"] + \
         config_param['choice_prob']['temper']["x0"]
    
    bounds = config_param['discount_func'][dstyle]["bound"] + \
             config_param['utility_func'][ustyle]["bound"] + \
             config_param['choice_prob']['temper']["bound"]

    if method == 'probit':
        np.random.seed(2023)
        regenerate_sample = False
        kwargs = {'simu_sample': np.random.normal(size=len(data)*simu_size).reshape(len(data),simu_size)}

        minimizer_kwargs = {"method": "L-BFGS-B", "args": (data, dstyle, ustyle, bounds, method,
                                                           regenerate_sample,kwargs)}
    else:
        minimizer_kwargs = {"method": "L-BFGS-B", "args": (data, dstyle, ustyle, bounds, method)}

    solver = basinhopping(log_likelihood, x0, minimizer_kwargs=minimizer_kwargs, 
                niter=100, 
                stepsize=0.05, 
                T=1.0,
                niter_success = 10,
                disp=disp_step)
    
    if solver.success:
        result = solver.lowest_optimization_result
        se = np.sqrt(np.diag(result.hess_inv.todense())) / np.sqrt(len(data))
        log_like = -result.fun
        aic = 2*len(x0)-2*log_like
        bic = 2*np.log(len(data))*len(x0)-2*log_like
        gradient = result.jac


        output= {'model':dstyle+'-'+ustyle,
                'params':[round(e,3) for e in result.x],
                'se':[round(e,3) for e in se],
                'gradient':[round(e,3) for e in gradient],
                'log-likelihood':round(log_like,3),
                'aic':round(aic,3),
                'bic':round(bic,3),
                }
        
        if disp_output:
            print(output)

        return output
    else:
        output = "Fail to converge"
        print(output)
        return output

    
#if __name__ == "__main__":

    


    


# estimation with Bayesian method

