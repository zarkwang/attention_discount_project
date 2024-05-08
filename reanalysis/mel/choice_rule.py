
import numpy as np
import inspect
import scipy.stats as st
from typing import Union
from mel import discount



class utility:

    def __init__(self,ss_x,ss_t,ll_x,ll_t,
                 params: Union[dict,list],
                 dstyle='expo',ustyle='power',
                 random=None,intercept=False):
        
        self.ss_x = ss_x #array or float, small reward
        self.ss_t = ss_t #array or float, small delay
        self.ll_x = ll_x #array or float, large reward
        self.ll_t = ll_t #array or float, large delay
        self._params = params
        self._dstyle= dstyle  
        self._ustyle = ustyle  
        self._random = random  
        self._intercept = intercept
        
        self.d_keys = self.get_dargs_keys(dstyle)
        self.u_keys = self.get_uargs_keys(required_param=['coef'],random=random,intercept=intercept)

        """
        params: dict or list, parameters in utility and discounting (timing cost) functions

            If paramters are specified in a list, then their order must be in line with 'd_keys'
            at first and 'u_keys' at second.
            
        ustyle: str, style of utility function, default is "power"
        dstyle: str, style of discounting function, default is "expo"
        random: can be None, "normal" or "gumbel"

                Default is None. If it is not None, a zero-mean normal-/Gumbel-distributed random 
                error will be added to utility. The scale parameter for the error's distribution 
                is represented by "u_scale".
            
        intercept: Boolean

                Default is False. When it is an intercept, represented by parameter "u_intercept"
                will be added to utility. 

        d_keys: keys for "dargs" in the chosen discounting (time cost) function
        u_keys: keys for "uargs" in the chosen utility function 
        """


        self.valid_dstyle = discount.d_func_list
        self.valid_ustyle = ['cara','power']
        self.d_args,self.u_args = self.get_all_args()

        self.diff = self.agg_utility()

        

    def instant_utility(self, uargs:dict, x):

        """
        Get the instantaneous utility for receiving a reward.

            x: amount of reward, array or float
            uargs: arguments in the utility function

                The parameters in "uargs" need to be specified in {key:value} form, where value 
                needs to be a float. All the valid keys are specified in "required_param". In
                addition, when "random" is not None, a scale parameter "u_scale" needs to be
                specified for random error; when "intercept" is True, a intercept parameter 
                "u_intercept" needs to be specified.
              
                For example, for power utility with a random error but no intercept, we can set "uargs" as 
                {"coef":0.8,"u_scale":1.0}.

            Supported styles for utility function:
                1.exponential (cara) 
                2.power (power)
                3.double power (power2)
                4.logarithmic (log)
        
        """

        if self._ustyle == 'cara':
            u = 1 - np.exp(- x * uargs[self.u_keys[0]])
        elif self._ustyle == 'power':
            u = x**uargs[self.u_keys[0]]
        else:
            print("Invalid value for argument 'ustyle'. Must be one of",self.valid_ustyle)

        if not self._random:
            utility = u
        elif self._random == 'normal':
            utility = u + np.random.normal(loc=0, scale=uargs['u_scale'], size= len(x))
        elif self._random == 'gumbel':
            utility = u + np.random.gumbel(loc=0, scale=uargs['u_scale'], size= len(x))
        else:
            print("Invalid value for argument 'random'. Must be one of 'normal', 'gumbel' or None.")

        if not self._intercept:
            return utility
        else:
            return utility + uargs['u_intercept']


    def discount_factor(self, dargs:dict, t, u):

        """
        Get the discounting factor for a reward that is received after a certain delay.

            t: delay
            u: instantaneous utility
            dargs: arguments in the discounting function
                
                The parameters in "dargs" need to be specified in {key:value} form, where value 
                needs to be a float. Each discounting function style has different keys for "dargs". 
                For example, for quasi-hyperbolic discounting, we can set "dargs" as 
                {"delta":0.8,"beta":0.9}.
        """

        if self._dstyle != 'trade':
            return eval('discount.'+self._dstyle+'_d')(dargs=dargs,t=t,u=u)
        else:
            print("'trade' is not a type of discounting factor. Please specify another 'dstyle'")


    @staticmethod
    def get_dargs_keys(dstyle):
        """
        Get the keys for the "dargs" in a discounting function
        """

        if dstyle not in discount.d_func_list:
            print("'dstyle' must be one of",discount.d_func_list)
        else:
            d_func = eval("discount."+dstyle+'_d')

        sig = inspect.signature(d_func)
        
        d_keys = sig.parameters['dargs'].annotation.__args__[0].__args__

        return list(d_keys)
    
    @staticmethod
    def get_uargs_keys(required_param,random,intercept):
        """
        Get the keys for the "uargs" in an instantaneous utility function
        """
        
        return required_param + intercept*['u_intercept'] + (random is not None)*['u_scale']
    

    def get_all_args(self):
        """
        Recover 'uarg' and 'dargs' from preset parameters
        """

        if isinstance(self._params,np.ndarray) or isinstance(self._params,list):
            
            d_args = dict(zip(self.d_keys,self._params[:len(self.d_keys)]))
            u_args = dict(zip(self.u_keys,self._params[len(self.d_keys):]))

        elif isinstance(self._params,dict):
            d_args = self._params[self.d_keys]
            u_args = self._params[self.u_keys]
        
        return d_args,u_args
    

    def discount_u(self,x,t):

        if self._dstyle != 'trade':

            u = self.instant_utility(self.u_args,x)
            d = self.discount_factor(self.d_args,t,u)
        
        else:
            print("Please check if 'dstyle' is suitable for discounted utility.")
        
        return d*u
    

    def agg_utility(self):
        
        """
        Aggregate the utilities for SS and LL rewards. 
        
        When 'dstyle' is 'trade', the result is the instantaneous utility of LL reward minus 
        that of SS reward, then minus the time cost. Otherwise, the result is the discounted 
        utility of LL reward minus that of SS reward.
        """

        if self._dstyle != 'trade':

            return self.discount_u(self.ll_x,self.ll_t) - self.discount_u(self.ss_x,self.ss_t)
        
        else:

            diff_u = self.instant_utility(self.u_args,self.ll_x) - \
                            self.instant_utility(self.u_args,self.ss_x)

            return diff_u - discount.trade_d(self.d_args,ss_t=self.ss_t,ll_t=self.ll_t)
    


# The choice probability for option LL
def choice_prob(ss_x, ss_t, ll_x, ll_t, params, dstyle, ustyle, temper,
                   intercept=False,
                   method="logit", 
                   regenerate_sample=True, simu_size=1000, random_state=2023,
                   **kwargs):

    diff_u = utility(ss_x,ss_t,ll_x,ll_t,params,dstyle,ustyle,intercept=intercept).diff

    method_list = ['logit','probit','binary']

    if method == "logit":
        p_choice_ll = 1/(1+np.exp(-diff_u/temper))
    
    elif method == "binary":
        p_choice_ll = (diff_u>=0)

    elif method == "probit":

        np.random.seed(random_state)

        if hasattr(ll_x, "__len__"):

            obs = len(ll_x)

            if regenerate_sample:
                simu_normal = np.random.normal(size=obs*simu_size).reshape(obs,simu_size)
            else:
                simu_normal = kwargs['simu_sample']
            
            diff_v = np.repeat(diff_u,simu_size).reshape(obs,simu_size) 

            cum_limit = (diff_v + simu_normal * temper) / temper

            p_choice_ll = np.mean(st.norm.cdf(cum_limit),axis=1)

        else:

            if regenerate_sample:
                simu_normal = np.random.normal(size=simu_size)
            else:
                simu_normal = kwargs['simu_sample']

            cum_limit = (diff_u + simu_normal * temper) / temper
            
            p_choice_ll = (st.norm.cdf(cum_limit)).mean() 

    else:
        print("Invalid method. Should be one of",method_list)
    
    return p_choice_ll



