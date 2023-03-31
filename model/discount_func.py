
import numpy as np
from typing import Dict, Union, Literal


"""
Discounting functions:
    1. exponential (expo)
    2. double exponential (expo2)
    3. hyperbolic (hb)
    4. dual-parameter hyperbolic (hb2)
    5. magnitude-dependent hyperbolic (hbmd)
    6. quasi-hyperbolic (quasihb)
    7. quasi-hyperbolic plus fixed delay cost (quasihb_fc)
    8. homogeneous costly empathy (hce)
    9. attention-adjusted (attention)
"""

# exponential discounting
def expo_discount(t,u,dargs: Dict[Literal["delta"], float]):

    w_t = dargs['delta']**t

    return w_t

# double exponential discounting (Van den Bos-McClure 2013)
def expo2_discount(t,u,dargs: Dict[Literal["delta1","delta2","d_weight"], float]):

    w_t = dargs['d_weight']*dargs['delta1']**t + (1-dargs['d_weight'])*dargs['delta2']**t

    return w_t

# hyperbolic discounting
def hb_discount(t,u,dargs: Dict[Literal["k"], float]):
    
    w_t = 1 / (1 + dargs['k'] * t)
    
    return w_t

# dual-parameter hyperbolic discounting (Loewenstein-Prelec 1992)
def hb2_discount(t,u,dargs: Dict[Literal["k","a"], float]):
    
    w_t = 1/(1 + dargs['k'] * t)**dargs['a']
    
    return w_t

# magnitude-dependent discounting (Gershman-Bhui 2020)
def hbmd_discount(t,u,dargs: Dict[Literal["b"], float]):

    k = 1 / (dargs['b'] * abs(u))

    w_t = 1 / (1 + k * t)
    
    return w_t

# quasi-hyperbolic discounting (Laibson 1997)
def quasihb_discount(t,u,dargs: Dict[Literal["beta","delta"], float]):
    
    w_t = (t==0) + (t!=0) * dargs['beta'] * dargs['delta']**t
    
    return w_t


# quasi-hyperbolic-plus-fixed-delay-cost discounting (Benhabib-Bisin-Schotter 2010)
def quasihb_fc_discount(t,u,dargs: Dict[Literal["beta","delta","c"], float]):
    
    w_t = (t==0) + (t!=0) * dargs['beta'] * dargs['delta']**t - dargs['c'] / u
    
    return w_t


# homogeneous costly empathy discounting (Noor-Takeoka 2022)
def hce_discount(t,u,dargs: Dict[Literal["delta","m"], float]):

    a_t = dargs['delta']**t

    w_t = a_t * u**(1/(dargs['m']-1)) 

    return w_t


def hce_bar_discount(t,u,dargs: Dict[Literal["delta","m","u_bar","d_bar"], float]):

    a_t = dargs['delta']**t

    w_t = a_t * u**(1/(dargs['m']-1)) * (u<dargs['u_bar']) + dargs['d_bar'] * (u>=dargs['u_bar'])

    return w_t


# attention-adjusted discounting
def attention_discount(t,u,dargs: Dict[Literal["delta","lambd"], float]):
    
    if dargs['delta'] == 1:
        g = t
    else:
        g = 1 / (1 - dargs['delta']) * (dargs['delta']**(-t) - 1)
    
    w_t = 1 / (1 + g * np.exp(-u / dargs['lambd']))
    
    return w_t



def attention_uni_discount(t,u,dargs: Dict[Literal["lambd"], float]):
    
    w_t = 1 / (1 + t * np.exp(-u / dargs['lambd']))
    
    return w_t

