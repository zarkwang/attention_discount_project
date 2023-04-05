
import numpy as np
import inspect
from typing import Dict, Union, Literal


"""
Supported Styles for Discounting / Time Cost Function:
    1. exponential (expo)
    2. double exponential (expo2)
    3. hyperbolic (hb)
    4. dual-parameter hyperbolic (hb2)
    5. magnitude-dependent hyperbolic (hbmd)
    6. quasi-hyperbolic (quasihb)
    7. quasi-hyperbolic plus fixed delay cost (quasihb_fc)
    8. homogeneous costly empathy (hce)
    9. attention-adjusted (attention)
    10. attention with uniform initial weight allocation (attention_uni)
    11. time cost in trade-off model (trade)
"""

d_func_list = ["expo","expo2",
                "hb","hb2","hbmd",
                "quasihb","quasihb_fc",
                "hce",
                "attention","attention_uni",
                "trade"]


# exponential discounting  
def expo_d(dargs: Dict[Literal["delta"], float],
                t=None,u=None):

    return dargs['delta']**t


# double exponential discounting (Van den Bos-McClure 2013)
def expo2_d(dargs: Dict[Literal["delta1","delta2","d_weight"], float],
                t=None,u=None):

    return dargs['d_weight']*dargs['delta1']**t + (1-dargs['d_weight'])*dargs['delta2']**t


# hyperbolic discounting
def hb_d(dargs: Dict[Literal["k"], float],
                t=None,u=None):
    
    return 1 / (1 + dargs['k'] * t)


# dual-parameter hyperbolic discounting (Loewenstein-Prelec 1992)
def hb2_d(self,dargs: Dict[Literal["k","a"], float],
                t=None,u=None):
    
    return 1/(1 + dargs['k'] * self._t)**dargs['a']

# magnitude-dependent discounting (Gershman-Bhui 2020)
def hbmd_d(dargs: Dict[Literal["b"], float],
                t=None,u=None):

    k = 1 / (dargs['b'] * abs(u))
    
    return 1 / (1 + k * t)

# quasi-hyperbolic discounting (Laibson 1997)
def quasihb_d(dargs: Dict[Literal["beta","delta"], float],
                    t=None,u=None):

    d = (t==0) + (t!=0) * dargs['beta'] * dargs['delta']**t
    
    return d


# quasi-hyperbolic-plus-fixed-delay-cost discounting (Benhabib-Bisin-Schotter 2010)
def quasihb_fc_d(dargs: Dict[Literal["beta","delta","c"], float],
                        t=None,u=None):

    d = (t==0) + (t!=0) * dargs['beta'] * dargs['delta']**t
    
    return  d - dargs['c'] / u


# homogeneous costly empathy discounting (Noor-Takeoka 2022)
def hce_d(dargs: Dict[Literal["delta","m"], float],
                t=None,u=None):

    a_t = dargs['delta']**t

    w_t = a_t * u**(1/(dargs['m']-1)) 

    return w_t


# attention-adjusted discounting
def attention_d(dargs: Dict[Literal["delta","lambd"], float],
                    t=None,u=None):
    
    if dargs['delta'] == 1:
        g = t
    else:
        g = 1 / (1 - dargs['delta']) * (dargs['delta']**(-t) - 1)
    
    w_t = 1 / (1 + g * np.exp(- u / dargs['lambd']))
    
    return w_t


# attention with uniform initial weight allocation
def attention_uni_d(dargs: Dict[Literal["lambd"], float],
                        t=None,u=None):
    
    g = t
    w_t = 1 / (1 + g * np.exp(- u / dargs['lambd']))
    
    return w_t



def trade_d(dargs: Dict[Literal["beta","zeta","alpha","k"],float],
            ss_t,ll_t):
    
    """
    Calculate the time cost in intertemporal trade-off model (Scholten & Read, 2010; 
    Scholten, Read & Sanborn, 2014)
    """

    w = lambda t: 1/dargs['beta'] * np.log(1+dargs['beta']*t)

    represent_t = ((w(ll_t) - w(ss_t)) / dargs['zeta'])**dargs['zeta']

    cost = dargs['k']/dargs['alpha'] * np.log(1+represent_t)

    return cost   



