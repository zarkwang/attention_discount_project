import yaml
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize,basinhopping
from sklearn import metrics
from mpl import choice_rule


with open('mpl/config_param.yaml', 'r') as f:
        config_param = yaml.safe_load(f)



class mle:

    def __init__(self,style,data,x0=None,bounds=None):
        
        self._data = data
        self._style = style
        self.dstyle = style['dstyle']
        self.ustyle = style['ustyle']
        self.method = style['method']
        self.intercept = style['intercept']

        init_intercept = [0]
        bound_intercept = [[-100,100]]

        if x0 is None:
            self._x0 = config_param['discount_func'][self.dstyle]["x0"] + \
                config_param['utility_func'][self.ustyle]["x0"] + \
                init_intercept*self.intercept + \
                config_param['choice_prob']['temper']["x0"]
        else:
            self._x0 = x0
        
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

        if bound:
            cons = ((np.array(bound)[:,0] <= params) & (np.array(bound)[:,1] >= params)).prod()

        return cons


    def objective(self,params,regenerate_sample=True,kwargs={}):

        if not self.constraint(params,self.bounds):
            return 1e10 
        else:
            temper = params[-1]
            ss_t = self._data['ss_t'].values
            ss_x = self._data['ss_x'].values
            ll_t = self._data['ll_t'].values
            ll_x = self._data['ll_x'].values
            choice = self._data['choice'].values
            
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

    
#if __name__ == "__main__":

    

