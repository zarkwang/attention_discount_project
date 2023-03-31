
import numpy as np
import inspect
from typing import Union
import discount_func


def get_dargs_keys(d_func):

    sig = inspect.signature(d_func)
    
    d_keys = sig.parameters['dargs'].annotation.__args__[0].__args__

    return list(d_keys)


def get_uargs_keys(ustyle,random):

    required_param = {'cara': ['coef'],
                      'power':['coef'],
                      'power2':['coef1','coef2'],
                      'log':['coef1','coef2']}

    if random:
        param_key = required_param[ustyle] + ['u_sigma']
    else:
        param_key = required_param[ustyle]
    
    return param_key



# instantaneous utility
def utility(x, uargs: dict, ustyle='power', random=False):
    
    param_key = get_uargs_keys(ustyle,random)

    if ustyle == 'cara':
        v = 1 - np.exp(-x * uargs[param_key[0]])
    elif ustyle == 'power':
        v = x**uargs[param_key[0]]
    elif ustyle == 'power2':
        v = x**uargs[param_key[0]]+uargs[param_key[1]]*x
    elif ustyle == 'log':
        v = uargs[param_key[0]] + uargs[param_key[1]]*np.log(x)
    else:
        print("Invalid value for argument 'ustyle'. Must be one of 'cara','power' or 'power2'.")

    if random:
        return v + np.random.normal(loc=0, scale=uargs['u_sigma'], size=1)
    else:
        return v



# discounted utility for each option in MPL paradigm
def mpl_du(t, x, params: Union[dict,list], 
          dstyle='attention', ustyle='power', random=False):
       
    # available discounting functions
    d_func_list = ["expo","expo2",
                   "hb","hb2","hbmd",
                   "quasihb","quasihb_fc",
                   "hce",
                   "attention","attention_uni"]

    # get discounting function
    if dstyle not in d_func_list:
        print("'dstyle' must be one of",d_func_list)
    else:
        d_func = eval('discount_func.'+dstyle+'_discount')

    # get dargs and uargs for each function
    d_keys = get_dargs_keys(d_func) 
    u_keys = get_uargs_keys(ustyle,random)

    if isinstance(params,np.ndarray) or isinstance(params,list):
        d_args = {d_keys[i]: params[i] for i in range(len(d_keys))}
        u_args = {u_keys[i]: params[i+len(d_keys)] for i in range(len(u_keys))}

    elif isinstance(params,dict):
        d_args = params[d_keys]
        u_args = params[u_keys]


    # calculate discounted utility
    u = utility(x, u_args, ustyle, random)
    du = d_func(t, u, d_args) * u
    return du


    


def choice_prob_du(ss_t, ss_x, ll_t, ll_x, params, dstyle, ustyle, temper=1):

    ss_v = mpl_du(ss_t,ss_x,params,dstyle,ustyle)
    ll_v = mpl_du(ll_t,ll_x,params,dstyle,ustyle)
    p_choice_ll = 1/(1+np.exp((ss_v - ll_v)/temper))
    
    return p_choice_ll


def choice_prob_itch(ss_t, ss_x, ll_t, ll_x, params):

    abs_diff_x = ll_x - ss_x
    abs_diff_t = ll_t - ss_t
    rel_diff_x = abs_diff_x / (ll_x + ss_x)
    rel_diff_t = abs_diff_t / (ll_t + ss_t)

    itch_list = [1,abs_diff_x,abs_diff_t,rel_diff_x,rel_diff_t] 

    p_choice_ll = sum([params[i] * itch_list[i] for i in range(len(itch_list))])

    return p_choice_ll

