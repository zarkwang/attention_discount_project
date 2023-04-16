import yaml
import os
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize,basinhopping
from sklearn import metrics
from mel import choice_rule


"""
    Load the configuration file. The following information are specified in configuration file:
    
    1. initial points (x0) and constraints (bound) for parameters in discounting function, utility
    function, and choice probability function
        - both the initial points and contraints are framed in list
        - e.g. for x0 = [.5,1,2], 0.5 is the initial point for the first parameter, 1 is the initial
        point for the second parameter, and so on 
        - for bound = [[.001,1],[.001,100]], the first parameter has a lower bound of 0.001 and a upper
        bound of 1, the second parameter has a lower bound of 0.001 and a upper bound of 100
    
    2. error message
"""

current_dir = os.path.dirname(os.path.relpath(__file__))
path_config = current_dir + '/config_param.yaml'

with open(path_config, 'r') as f:
        config_param = yaml.safe_load(f)



"""
    mle (maximum likelihood estimation)

    input:
        data: must be a dataframe and contain these five columns [ss_x,ss_t,ll_x,ll_t,choice] 
        style: model style, need to write in dict, such as 

            {'dstyle':<discount_function_name>,
            'ustyle': <utility_function_name>,
            'method':'logit' or 'probit',
            'intercept': True or False} 

        x0: initial points for optimization. if None, then use the initial points in the configuration file
        bounds: constraints. if None, the use the bounds in the figuration file 

    output:
        solver: solver in scipy.basinhopping function
        params: fitted parameters, listed in the following order 
            
            [<discount_parameters>,<utility_parameter>,<choice_probability_parameter>]
        
        se: standard error for each parameter
        gradient: gradient for each parameter
        log_loss: log loss, the objective for optimization
        aic: Arkaike Information Criterion
        bic: Bayesian Information Criterion
        output: summarizing the parameters and scores in a dict 

"""

class mle:

    def __init__(self,style:dict,data,x0=None,bounds=None):
        
        self._data = data
        self._style = style

        self.dstyle = style['dstyle']
        self.ustyle = style['ustyle']
        self.method = style['method']
        self.intercept = style['intercept']

        init_intercept = [0]
        bound_intercept = [[-100,100]]

        # Specify initial points (x0)
        if x0 is None:
            self._x0 = config_param['discount_func'][self.dstyle]["x0"] + \
                config_param['utility_func'][self.ustyle]["x0"] + \
                init_intercept*self.intercept + \
                config_param['choice_prob']['temper']["x0"]
        else:
            self._x0 = x0


        # Specify parameter constraints (bound)
        if bounds is None: 
            self.bounds = config_param['discount_func'][self.dstyle]["bound"] + \
                    config_param['utility_func'][self.ustyle]["bound"] + \
                    bound_intercept*self.intercept + \
                    config_param['choice_prob']['temper']["bound"]
        else:
            self.bounds = bounds
        
    

    @staticmethod
    def constraint(params,bound):
        cons = True

        # Judge if the value of every parameter is within the bounds
        if bound:
            cons = ((np.array(bound)[:,0] <= params) & (np.array(bound)[:,1] >= params)).prod()

        return cons


    def objective(self,params,regenerate_sample=True,kwargs={}):

        if not self.constraint(params,self.bounds):
            return 1e10 
        # If any parameter value is out of the bounds, then loss function will jump to a very high value
        else:
            temper = params[-1]
            ss_t = self._data['ss_t'].values
            ss_x = self._data['ss_x'].values
            ll_t = self._data['ll_t'].values
            ll_x = self._data['ll_x'].values
            choice = self._data['choice'].values
            
            # Compute the probability of choosing option LL
            if not regenerate_sample:
                p_choice_ll = choice_rule.choice_prob(ss_x, ss_t, ll_x, ll_t, params, 
                                                    dstyle=self.dstyle, 
                                                    ustyle=self.ustyle, 
                                                    temper=temper, 
                                                    intercept=self.intercept, 
                                                    method=self.method, 
                                                    regenerate_sample=False, 
                                                    simu_sample=kwargs['simu_sample'])
            else:
                p_choice_ll = choice_rule.choice_prob(ss_x, ss_t, ll_x, ll_t, params,
                                                    dstyle=self.dstyle, 
                                                    ustyle=self.ustyle, 
                                                    temper=temper, 
                                                    intercept=self.intercept,
                                                    method=self.method)
            
            p_choice_ll[np.where(p_choice_ll == 1)] = p_choice_ll[np.where(p_choice_ll == 1)] - 1e-8
            p_choice_ll[np.where(p_choice_ll == 0)] = p_choice_ll[np.where(p_choice_ll == 0)] + 1e-8

            loss = metrics.log_loss(choice,p_choice_ll)

            return loss
        
        
    def solve(self,niter=500,niter_sucess=50,
              regenerate_sample=True,simu_size=1000,
              disp_output=False,disp_step=False,):
        
        if self.method == 'probit':
            np.random.seed(2023) 
            regenerate_sample = False

            # Use Monte Carlo method to solve the multinominal probir model
            simu_sample = np.random.normal(size=len(self._data)*simu_size).reshape(len(self._data),simu_size)
            kwargs = {'simu_sample':simu_sample}

            minimizer_kwargs = {"method": "L-BFGS-B", 
                                "args": (regenerate_sample, kwargs)}
        else:
            minimizer_kwargs = {"method": "L-BFGS-B"}
            
        #stepsize = np.array(self.bounds)[:,1]*0.1

        solver = basinhopping(self.objective, self._x0, 
                              minimizer_kwargs=minimizer_kwargs, 
                              T=1.0,
                              niter=niter, 
                              stepsize=0.5, 
                              niter_success=niter_sucess,
                              disp=disp_step)
        
        self.solver = solver
        #print(solver)
        
        if solver.success:
            result = solver.lowest_optimization_result
            self.params = result.x
            self.se = np.sqrt(np.diag(result.hess_inv.todense())) / np.sqrt(len(self._data))
            self.log_loss = result.fun
            self.aic = 2*len(self._x0)+2*self.log_loss*len(self._data)
            self.bic = 2*np.log(len(self._data))*len(self._x0)+2*self.log_loss*len(self._data)
            self.gradient = result.jac

            self.output= {'style':[self._style],
                        'params':[round(e,3) for e in self.params],
                        'se':[round(e,3) for e in self.se],
                        'gradient':[round(e,3) for e in self.gradient],
                        'log_loss':round(self.log_loss,3),
                        'aic':round(self.aic,3),
                        'bic':round(self.bic,3)
                        }
            
        else:
            self.output = config_param['msg']['fail_converge']
            self.params = solver.lowest_optimization_result.x
        
        if disp_output:
                print(self.output)

        return self.output






def gen_style_list(method='logit',intercept=False):

    """
    Find the discounting functions and utility functions specified in the configuration file, then
    generate a list of model styles (in dictionary form)
    """

    dstyle_list = list(config_param['discount_func'].keys())
    ustyle_list = list(config_param['utility_func'].keys())

    style_list = [ {"dstyle":dstyle_list[i], 
                    "ustyle":ustyle_list[j], 
                    "method":method, 
                    "intercept":intercept} 
                for i in range(len(dstyle_list)) for j in range(len(ustyle_list))
                ]
    
    return style_list


    
#if __name__ == "__main__":

    

