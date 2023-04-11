import yaml
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize,basinhopping
from sklearn import metrics
from mpl import choice_rule


with open('mpl/config_param.yaml', 'r') as f:
        config_param = yaml.safe_load(f)


def constraint(params,bound):
    cons = True

    if bound:
        cons = ((np.array(bound)[:,0] <= params) & (np.array(bound)[:,1] >= params)).prod()

    return cons



def objective(params, data, dstyle, ustyle, method="logit", 
                intercept=False,
                bounds=None,
                regenerate_sample=True,
                kwargs={}):

    if not constraint(params,bounds):
        return 1e10 
    else:
        temper = params[-1]
        ss_t = data['ss_t'].values
        ss_x = data['ss_x'].values
        ll_t = data['ll_t'].values
        ll_x = data['ll_x'].values
        choice = data['choice'].values
        
        if not regenerate_sample:
            p_choice_ll = choice_rule.choice_prob(ss_x, ss_t, ll_x, ll_t, 
                                                  params, dstyle, ustyle, temper, 
                                                  intercept=intercept, 
                                                  method=method, 
                                                  regenerate_sample=False, 
                                                  simu_sample=kwargs['simu_sample'])
        else:
            p_choice_ll = choice_rule.choice_prob(ss_x, ss_t, ll_x, ll_t, 
                                                  params, dstyle, ustyle, temper, 
                                                  intercept=intercept,
                                                  method=method)
        
        p_choice_ll[np.where(p_choice_ll == 1)] = p_choice_ll[np.where(p_choice_ll == 1)] - 1e-8
        p_choice_ll[np.where(p_choice_ll == 0)] = p_choice_ll[np.where(p_choice_ll == 0)] + 1e-8

        loss = metrics.log_loss(choice,p_choice_ll)

        return loss





def mle(style,data,disp_output=False,disp_step=False,simu_size=1000):

    dstyle = style['dstyle']
    ustyle = style['ustyle']
    method = style['method']
    intercept = style['intercept']

    init_intercept = [0]
    bound_intercept = [[-100,100]]

    x0 = config_param['discount_func'][dstyle]["x0"] + \
         config_param['utility_func'][ustyle]["x0"] + \
         init_intercept*intercept + \
         config_param['choice_prob']['temper']["x0"]
    
    bounds = config_param['discount_func'][dstyle]["bound"] + \
             config_param['utility_func'][ustyle]["bound"] + \
             bound_intercept*intercept + \
             config_param['choice_prob']['temper']["bound"]

    if method == 'probit':
        np.random.seed(2023)
        regenerate_sample = False
        kwargs = {'simu_sample': np.random.normal(size=len(data)*simu_size).reshape(len(data),simu_size)}

        minimizer_kwargs = {"method": "L-BFGS-B", 
                            "args": (data, dstyle, ustyle, method, intercept, bounds, 
                                     regenerate_sample, kwargs)}
    else:
        minimizer_kwargs = {"method": "L-BFGS-B", 
                            "args": (data, dstyle, ustyle, method, intercept, bounds)}
        
    
    stepsize = np.array(bounds)[:,1]*0.1

    solver = basinhopping(objective, x0, minimizer_kwargs=minimizer_kwargs, 
                niter=1000, 
                stepsize=stepsize, 
                T=1.0,
                niter_success = 100,
                disp=disp_step)
    
    #print(solver)
    
    if solver.success:
        result = solver.lowest_optimization_result
        se = np.sqrt(np.diag(result.hess_inv.todense())) / np.sqrt(len(data))
        log_loss = result.fun
        aic = 2*len(x0)+2*log_loss*len(data)
        bic = 2*np.log(len(data))*len(x0)+2*log_loss*len(data)
        gradient = result.jac


        output= {'model':dstyle+'-'+ustyle,
                'params':[round(e,3) for e in result.x],
                'se':[round(e,3) for e in se],
                'gradient':[round(e,3) for e in gradient],
                'log_loss':round(log_loss,3),
                'aic':round(aic,3),
                'bic':round(bic,3),
                }
        
        if disp_output:
            print(output)

        return output
    else:
        output = config_param['msg']['fail_converge']
        print(output)
        return output

    
#if __name__ == "__main__":

    

