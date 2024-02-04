import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import multiprocessing as mp 
from scipy import stats
from tqdm import tqdm


def median_plot(data,label_name):

        # calculate median answer for each condition of the data
        df_plot_median = data.groupby(['seq_length','front_amount',label_name])['value_surplus'].median().to_frame().reset_index()
        front_amount_list = df_plot_median['front_amount'].unique()
        seq_length_list = df_plot_median['seq_length'].unique()

        # configure color and linewidth
        color_list = ['c','m']
        linewidth_list = [4,3]

        # create a figure
        fig,ax  = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={'height_ratios': [0.15, 0.85]})

        # draw lines
        for t in range(len(seq_length_list)):
                tab_plot_0 = df_plot_median[(df_plot_median['seq_length'] == seq_length_list[t]) & (df_plot_median[label_name] == 0)]
                tab_plot_1 = df_plot_median[(df_plot_median['seq_length'] == seq_length_list[t]) & (df_plot_median[label_name] == 1)]
                ax[1].plot(tab_plot_0['front_amount'],tab_plot_0['value_surplus'],ls='solid',c=color_list[t],lw=linewidth_list[t])
                ax[1].plot(tab_plot_1['front_amount'],tab_plot_1['value_surplus'],ls='dashed',c=color_list[t],lw=linewidth_list[t])

        ax[1].plot(np.NaN,np.NaN,ls='solid',c='black')
        ax[1].plot(np.NaN,np.NaN,ls='dashed',c='black')

        # calculate the number of subjects in each cluster
        n_dot_cluster = np.bincount(data[label_name]) / len(data) * len(data['worker_id'].unique())
        cluster_1 = f"cluster 1 (N={int(n_dot_cluster[0])})"
        cluster_2 = f"cluster 2 (N={int(n_dot_cluster[1])})"

        # make legends
        lines = ax[1].get_lines()
        legend1 = plt.legend([lines[i] for i in [0,2]], ["12 months", "6 months"], title='sequence length',
                        loc='upper center', bbox_to_anchor=(0.3, 1.26))
        legend2 = plt.legend([lines[i] for i in [4,5]], [cluster_1, cluster_2], title='cluster',
                        loc='upper center', bbox_to_anchor=(0.62, 1.26))
        ax[1].add_artist(legend1)
        ax[1].add_artist(legend2)

        ax[0].axis('off')

        # add axis ticks and labels        
        plt.yticks(np.arange(20,70,step=5))
        plt.xticks(front_amount_list)
        plt.xlabel('Front-end amount (£)')
        plt.ylabel('Indifference point minus front-end amount (£)')
        plt.tight_layout()
        plt.show()


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

    def fit(self,maxiter=200):

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

        boots_rlm = sm.RLM(endog=boots_y,exog=boots_X,M=sm.robust.norms.HuberT())

        return boots_rlm.fit(maxiter=maxiter,scale_est=sm.robust.scale.HuberScale())
    

    def fit_func(self,i):
        return i,self.fit()
    

    def bootstrap(self,n_jobs=8):

        with mp.Pool(processes=n_jobs) as pool:

            _names = ['const'] + self.param_names

            result_dict = {p:[] for p in _names + ['scale','deviance','nobs','cond_loss','sample_id']}

            for i,result in tqdm(
                                pool.imap_unordered(self.fit_func,range(self.n_bootstrap)),
                                total=self.n_bootstrap):

                for param in _names:
                    try:
                        result_dict[param] += [result.params[param]]
                    except:
                        result_dict[param] += [np.NaN]
                
                try:
                    result_dict['scale'] += [result.scale]
                    result_dict['deviance'] += [result.fit_history['deviance'][-1]]
                    result_dict['nobs'] += [result.nobs]
                    result_dict['cond_loss'] += [self.muller_welsh_loss(result)]
                except:
                    result_dict['scale'] += [np.NaN]
                    result_dict['deviance'] += [np.NaN]
                    result_dict['nobs'] += [np.NaN]
                    result_dict['cond_loss'] += [np.NaN]
                                                 
                result_dict['sample_id'] += [i]

        pool.join()
        pool.close()

        self.bootstrapResult = pd.DataFrame(result_dict)

        try:
            self.mull_welsh_score = self.muller_welsh_criterion()
            self.ci = self.conf_int()
        except:
            self.mull_welsh_score = None
            self.ci = None

        return self.bootstrapResult


    def muller_welsh_criterion(self):
        
        term_1 = self.muller_welsh_loss(model=self.model)
        term_2 = self.model.nobs * len(self.model.params)
        term_3 = self.bootstrapResult['conditional_loss'].mean()

        criterion = self.model.scale / self.model.nobs * (term_1 + term_2 + term_3)
        
        return criterion
    
    
    def conf_int(self,sig=0.05):

        _names = ['const'] + self.param_names

        coef = self.bootstrapResult[_names].median() 
        se = self.bootstrapResult[_names].std() 
        mad = self.bootstrapResult[_names].apply(lambda x:np.median(np.abs(x - np.median(x))))
        ci_lower = self.bootstrapResult[_names].apply(lambda x:sorted(x)[int(sig/2 * len(self.bootstrapResult))])
        ci_upper = self.bootstrapResult[_names].apply(lambda x:sorted(x,reverse=True)[int(sig/2 * len(self.bootstrapResult))])
        
        return pd.DataFrame({'median_coef':coef,'se':se,'mad':mad,'ci_lower':ci_lower,'ci_upper':ci_upper})
    
    
    @staticmethod
    def muller_welsh_loss(model,b=2):

        loss = model.sresid.apply(lambda x:min(x**2,b**2)) @ model.weights

        return loss
    
    
    @staticmethod
    def stratified_sample(data,model=None,col_name=None,ratio_of_sample_size=0.5,
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
    
    




    

    
    


