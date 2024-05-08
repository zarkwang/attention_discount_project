import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize 
import statsmodels.api as sm


def utility(x,gamma,m):
    return (m+x)**gamma

def p_risk(x_risk,x_safe,gamma,temp,m):

    v_safe = utility(x_safe,gamma,m)
    v_risk = utility(x_risk,gamma,m)

    v = (v_safe - v_risk*0.5) / temp
    p = 1/(1+np.exp(-v))

    # p = (0.5*v_risk/v_safe)**(1/temp)

    return p

class utilLogit:

    def __init__(self,data):
        self.y = data['choice'] # choice = 1 if choose the safe option; = 0 if choose the risky option
        self.x_risk = data['risk_amount']
        self.x_safe = data['safe_amount']

    def logLike(self,param):

        gamma = param[0]
        temp = param[1]
        m = param[2]

        p = p_risk(self.x_risk,self.x_safe,gamma,temp,m)

        obj =  (1-self.y) @ np.log(1-p) + self.y @ np.log(p)

        return -obj
    

    def fit(self):
        
        x0 = [1,1,0]
        bounds = [(0, 1),(0,None),(0,None)]
        opt = optimize.minimize(self.logLike,x0,method='SLSQP',bounds=bounds)

        # opt = optimize.basinhopping(self.logLike,x0,
        #                             minimizer_kwargs={'method':'SLSQP','bounds':bounds},
        #                             stepwise_factor = 0.1, T = 0.5)

        return opt





