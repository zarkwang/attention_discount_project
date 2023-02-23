
import numpy as np


# instantaneous utility (random)
def utility(x, uargs, ustyle='crra', sigma=0):
    
    if ustyle == 'crra':
        v = np.power(x, uargs['rra'])
    elif ustyle == 'cara':
        v = 1 - np.exp(-x * uargs['ara'])
    else:
        print("Invalid value for argument 'ustyle'. Must be 'crra' or 'cara'.")
        return
    
    u = v + np.random.normal(loc=0, scale=sigma, size=1)
    
    return u


# attention-adjusted discounting function
def attention_mpl_d(t, u, dargs):
    
    if dargs['delta'] == 1:
        g = t
    elif dargs['delta'] < 1 and dargs['delta'] > 0:
        g = 1 / (1 - dargs['delta']) * (np.power(dargs['delta'], -t) - 1)
    else:
        print("Invalid value for argument 'delta'. Must be greater than 0 and not greater than 1.")
        return
    
    w_t = 1 / (1 + g * np.exp(-u / dargs['lambd']))
    
    return w_t


# hyperbolic discounting function
def hb_mpl_d(t, u, dargs):
    
    w_t = 1 / (1 + dargs['k'] * t)
    
    return w_t


# quasi-hyperbolic discounting function
def quasihb_mpl_d(t, u, dargs):
    
    w_t = (t == 0) + (t!=0) * dargs['beta'] * np.power(dargs['delta'], t)
    
    return w_t


# Loewenstein-Prelec discounting function
def lp_mpl_d(t, u, dargs):
    
    w_t = np.power(1 + dargs['k'] * t, -dargs['a'] / dargs['k'])
    
    return w_t


# Benhabib-Bisin-Schotter discounting function
def bbs_mpl_d(t, u, dargs):
    
    w_t = (t==0) + (t!=0) * np.exp(-dargs['k'] * t) - dargs['b'] / u
    
    return w_t


# Gershman-Bhui discounting function
def gb_mpl_d(t, u, dargs):
    
    k = 1 / (dargs['beta'] * abs(u))
    
    w_t = 1 / (1 + k * t)
    
    return w_t



# required parameters
d_required_params = {'hb': ['k'],
                'quasihb': ['beta','delta'],
                'lp': ['k','a'],
                'bbs': ['k','b'],
                'gb': ['beta'],
                'attention': ['delta','lambd']
               }

u_required_params = {'crra': ['rra'],
                    'cara': ['ara']     
                }



# discounted utility for each option in MPL paradigm
def mpl_v(t, x, params, dstyle, ustyle='crra', sigma=0, **kwargs):

    if isinstance(params,np.ndarray):
        require = d_required_params[dstyle] + u_required_params[ustyle]
        args = {require[i]: params[i] for i in range(len(require))} 
    elif isinstance(params,dict):
        args = params
    else:
        print("'params' can only be am array or a dict")
    
    u = utility(x, args, ustyle, sigma)
    
    fun_list = {'hb': hb_mpl_d,
                'quasihb': quasihb_mpl_d,
                'lp': lp_mpl_d,
                'bbs': bbs_mpl_d,
                'gb': gb_mpl_d,
                'attention': attention_mpl_d
               }
    
    if dstyle not in fun_list:
        print("Invalid value for argument 'dstyle'. Must be one of 'hb', 'quasihb', 'lp', 'bbs', 'gb' or 'attention'.")
        return
    
    d = fun_list[dstyle](t, u, args)
    du = d * u
    
    return du
