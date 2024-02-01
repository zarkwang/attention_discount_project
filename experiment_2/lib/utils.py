import numpy as np
import pandas as pd
import statsmodels.api as sm
import multiprocessing as mp 
from scipy import stats
from tqdm import tqdm


def alpha_outlier(data_array,scale_est='std',alpha_tilde=0.05,hampel_threshold=5.2,display=True):
    """
    alpha_tilde: bound of the probability of identifying at least one non-outlier observation 
     (assuming iid normal distribution) as outliers. Default = 0.05
    """

    if scale_est == 'std':
        _alpha = 1 - (1 - alpha_tilde)**(1/len(data_array))
        _threshold = stats.norm.ppf(1 - _alpha / 2)
        n_upper = ((data_array - np.mean(data_array)) / np.std(data_array) > _threshold).sum()
        n_lower = (- (data_array - np.mean(data_array)) / np.std(data_array) > _threshold).sum()    
    
    elif scale_est == 'mad':
        mad = np.abs(data_array - np.median(data_array)).median()
        n_upper = ((data_array - np.median(data_array)) / mad > hampel_threshold).sum()
        n_lower = (- (data_array - np.median(data_array)) / mad > hampel_threshold).sum()

    if display == True:

        print('Number of outliers (lower and upper):', [n_lower, n_upper])

    return n_upper, n_lower


def rule_out_outlier(data,col_name,**kwargs):
    
    n_upper, n_lower = alpha_outlier(data[col_name],**kwargs)
    
    upper_bound = sorted(data[col_name], reverse=True)[n_upper - 1]
    lower_bound = sorted(data[col_name])[n_lower - 1]
    
    data_filtered = data[(data[col_name] > lower_bound) & (data[col_name] < upper_bound)]

    return data_filtered


def get_reg_result(model,coef_names):
    
    result = {}
    _mod_cols = ['aic','bic','rsquared_adj','nobs']
    _coef_cols = ['params','tvalues','pvalues']

    for attr in _mod_cols:
        try:
            result[attr] = getattr(model,attr)
        except:
            if attr == 'aic':
                result[attr] = 2*(len(getattr(model,'params')) + getattr(model,'fit_history')['deviance'][-1])
            else:
                result[attr] = ''
            
    loc_coef_names = getattr(model,'params').index.isin(['const'] + coef_names)
    result['coef_name'] = getattr(model,'params').index[loc_coef_names].tolist()
    for attr in _coef_cols:
        try:
            result[attr] = getattr(model,attr)[loc_coef_names].tolist()
        except:
            result[attr] = ''
    
    ci_values = model.conf_int()[loc_coef_names]
    result['ci_lower'] = ci_values[0].tolist()
    result['ci_upper'] = ci_values[1].tolist()

    return result


class bootstrap_model:

    def __init__(self,data,model,param_names,n_bootstrap,fe=False):
        self.data = data
        self.model = model
        self.param_names = param_names
        self.n_bootstrap = n_bootstrap
        self.fe = fe

    def fit(self,maxiter=100):

        boots_sample = self.stratified_sample(self.data,model=self.model)

        if self.fe:
            sample = pd.concat([boots_sample,pd.get_dummies(boots_sample['worker_id'], prefix='worker_id')],
                               axis=1)
            endog_cols = self.param_names + [col for col in sample.columns if col.startswith('worker_id_')]
        else:
            sample = boots_sample
            endog_cols = self.param_names

        boots_y = sample['value_surplus']
        boots_X = sm.add_constant(sample[endog_cols]).astype(float)

        boots_rlm = sm.RLM(endog=boots_y,exog=boots_X,M=sm.robust.norms.HuberT()).fit(maxiter=maxiter)

        return boots_rlm
    

    def fit_func(self,i):
        return i,self.fit()
    

    def bootstrap(self,n_jobs=4):

        with mp.Pool(processes=n_jobs) as pool:

            _names = ['const'] + self.param_names

            result_dict = {p:[] for p in _names + ['scale','deviance','nobs','conditional_loss','sample_id']}

            for i,result in tqdm(
                                pool.imap_unordered(self.fit_func,range(self.n_bootstrap)),
                                total=self.n_bootstrap):

                for param in _names:
                    result_dict[param] += [result.params[param]]

                result_dict['scale'] += [result.scale]
                result_dict['deviance'] += [result.fit_history['deviance'][-1]]
                result_dict['nobs'] += [result.nobs]
                result_dict['conditional_loss'] += [self.muller_welsh_loss(result)]
                result_dict['sample_id'] += [i]

        pool.join()
        pool.close()

        return pd.DataFrame(result_dict)


    @staticmethod
    def muller_welsh_loss(model,b=2):

        loss = model.sresid.apply(lambda x:min(x**2,b**2)) @ model.weights

        return loss
    
    @staticmethod
    def stratified_sample(data,model=None,col_name=None,ratio_of_sample_size=0.3,
                      scale_est='mad',display=False,**kwargs):
    
        if model is not None:
            df_lower = data[(model.weights == 1) & (model.resid < model.resid.median())]
            df_upper = data[(model.weights == 1) & (model.resid > model.resid.median())]

            middle_index = set(data) - set(df_lower.index) - set(df_upper.index)

            df_middle = data[list(middle_index)]
        
        elif col_name is not None:
            n_upper, n_lower = alpha_outlier(data[col_name],scale_est=scale_est,display=display,**kwargs)

            upper_bound = sorted(data[col_name], reverse=True)[n_upper - 1]
            lower_bound = sorted(data[col_name])[n_lower - 1]

            df_lower = data[data[col_name] <= lower_bound]
            df_upper = data[data[col_name] >= upper_bound]
            df_middle = data[(data[col_name] > lower_bound) &
                            (data[col_name] < upper_bound)]
        
        n_lower = np.round(ratio_of_sample_size * len(df_lower)).astype(int)
        n_upper = np.round(ratio_of_sample_size * len(df_upper)).astype(int)
        n_middle = np.round(ratio_of_sample_size * len(df_middle)).astype(int)

        if display == True:
            print("observations drawn from each sample strata:", [n_lower,n_middle,n_upper])

        df_union = pd.concat([df_lower.sample(n=n_lower, replace=True), 
                            df_upper.sample(n=n_upper, replace=True), 
                            df_middle.sample(n=n_middle, replace=True)],ignore_index=True)


        return df_union




    

    
    


