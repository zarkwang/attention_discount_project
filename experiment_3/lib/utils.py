import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import multiprocessing as mp 
from scipy import stats
from tqdm import tqdm


def median_plot(data,label_name,method='median',hide_axis=False):

        # calculate median answer for each condition of the data
        if method == 'mean':
             df_plot_median = data.groupby(['seq_length','front_amount',label_name])['value_surplus'].mean().to_frame().reset_index()
        else:
            df_plot_median = data.groupby(['seq_length','front_amount',label_name])['value_surplus'].median().to_frame().reset_index()
        
        front_amount_list = df_plot_median['front_amount'].unique()
        seq_length_list = df_plot_median['seq_length'].unique()

        # configure color and linewidth
        color_list = ['c','m']
        linewidth_list = [4,3]

        # create a figure
        if hide_axis == False:
            fig_size = (9,8)
        else:
            fig_size = (8.5,8)
            
        fig,ax  = plt.subplots(2, 1, figsize=fig_size, gridspec_kw={'height_ratios': [0.2, 0.8]})

        # draw lines
        for t in range(len(seq_length_list)):
                tab_plot_0 = df_plot_median[(df_plot_median['seq_length'] == seq_length_list[t]) & (df_plot_median[label_name] == 0)]
                tab_plot_1 = df_plot_median[(df_plot_median['seq_length'] == seq_length_list[t]) & (df_plot_median[label_name] == 1)]
                ax[1].plot(tab_plot_0['front_amount'],tab_plot_0['value_surplus'],ls='dashed',c=color_list[t],lw=linewidth_list[t])
                ax[1].plot(tab_plot_1['front_amount'],tab_plot_1['value_surplus'],ls='solid',c=color_list[t],lw=linewidth_list[t])

        ax[1].plot(np.NaN,np.NaN,ls='dashed',c='black')
        ax[1].plot(np.NaN,np.NaN,ls='solid',c='black')

        # calculate the number of subjects in each cluster
        n_dot_cluster = np.bincount(data[label_name]) / len(data) * len(data['worker_id'].unique())
        cluster_1 = f"cluster 1 (N={int(n_dot_cluster[0])})"
        cluster_2 = f"cluster 2 (N={int(n_dot_cluster[1])})"

        # make legends
        if hide_axis == False:
            lines = ax[1].get_lines()
            legend1 = plt.legend([lines[i] for i in [0,2]], ["12 months", "6 months"], title='sequence length',
                            loc='upper center', bbox_to_anchor=(0.25, 1.35))
            legend2 = plt.legend([lines[i] for i in [4,5]], [cluster_1, cluster_2], title='cluster',
                            loc='upper center', bbox_to_anchor=(0.75, 1.35))
            ax[1].add_artist(legend1)
            ax[1].add_artist(legend2)


        ax[0].axis('off')

        # add axis ticks and labels 
        if hide_axis == False:       
            plt.ylabel('Indifference point minus front-end amount (£)')

        plt.xlabel('Front-end amount (£)')
        plt.yticks(np.arange(20,70,step=5))
        plt.xticks(front_amount_list)
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
    _mod_cols = ['aic','bic','nobs']
    _coef_cols = ['params','bse','tvalues','pvalues']

    for attr in _mod_cols:
        try:
            result[attr] = getattr(model,attr)
        except:
            if attr == 'aic':
                result[attr] = 2*(len(getattr(model,'params')) + getattr(model,'fit_history')['deviance'][-1])
            else:
                result[attr] = ''
            
    loc_coef_names = getattr(model,'params').index.isin(['const','Intercept'] + coef_names)
    result['coef_name'] = getattr(model,'params').index[loc_coef_names].tolist()
    for attr in _coef_cols:
        try:
            result[attr] = getattr(model,attr)[loc_coef_names].tolist()
        except:
            result[attr] = ''
    
    ci_values = model.conf_int()[loc_coef_names]
    result['ci_lower'] = ci_values[0].tolist()
    result['ci_upper'] = ci_values[1].tolist()

    dummies = model.params[['worker_id_' in i or 'C(worker_id)' in i for i in model.params.index]]
    result['contrast_mean'] = dummies.sum() / (len(dummies) +1)

    return result


class bootstrap_model:

    def __init__(self,data,model,param_names,n_bootstrap,fe=False,ratio_of_sample_size=0.5):
        self.data = data
        self.model = model
        self.param_names = param_names
        self.n_bootstrap = n_bootstrap
        self.fe = fe
        self.ratio_of_sample_size=ratio_of_sample_size

        if self.fe:
            self.data = pd.concat([self.data,pd.get_dummies(self.data['worker_id'], prefix='worker_id')],
                               axis=1)
            self.endog_cols = self.param_names + [col for col in self.data.columns if col.startswith('worker_id_')]
        else:
            self.endog_cols = self.param_names


    def fit(self,maxiter=200,bootstrap=True):

        if bootstrap == True:
            sample = self.stratified_sample(self.data,model=self.model,
                                                ratio_of_sample_size=self.ratio_of_sample_size)

            y = sample['value_surplus']
            X = sm.add_constant(sample[self.endog_cols]).astype(float)
        else:
            y = self.data['value_surplus']
            X = sm.add_constant(self.data[self.endog_cols]).astype(float)

        boots_rlm = sm.RLM(endog=y,exog=X,M=sm.robust.norms.HuberT())

        return boots_rlm.fit(maxiter=maxiter,scale_est=sm.robust.scale.HuberScale())
    

    def fit_func(self,i):
        try:
            result = self.fit()
        except:
            result = {key:np.NaN for key in ['scale','deviance','nobs','cond_loss']}
        return i,result
    

    def bootstrap(self,n_jobs=8):

        with mp.Pool(processes=n_jobs) as pool:

            _names = ['const'] + self.param_names

            result_dict = {p:[] for p in _names + ['scale','deviance','cond_loss','nobs','sample_id']}

            for i,result in tqdm(
                                pool.imap_unordered(self.fit_func,range(self.n_bootstrap)),
                                total=self.n_bootstrap):

                for param in _names:
                    try:
                        result_dict[param] += [result.params[param]]
                    except:
                        result_dict[param] += [np.NaN]
                
                try:
                    dev = result.fit_history['deviance'][-1]
                    cond_loss = self.get_cond_loss(result)
                except:
                    dev = result['deviance']
                    cond_loss = result['cond_loss']
                
                try:
                    result_dict['nobs'] += [result.nobs] 
                    result_dict['scale'] += [result.scale]
                    result_dict['deviance'] += [dev]
                    result_dict['cond_loss'] += [cond_loss]
                except:
                    for score in ['scale','deviance','cond_loss','nobs']:
                        result_dict[score] += [np.NaN]

                result_dict['sample_id'] += [i]

        pool.join()
        pool.close()

        self.bootstrapResult = pd.DataFrame(result_dict)


        return self.bootstrapResult


    def muller_welsh_criterion(self):
        
        term_1 = self.get_cond_loss(result=self.model)
        term_2 = 2*np.log(self.model.nobs) * len(self.model.params)
        term_3 = self.bootstrapResult['cond_loss'].mean()

        criterion = self.model.scale**2 / self.model.nobs * (term_1 + term_2 + term_3)
        
        return criterion
    
    
    def conf_int(self,sig=0.05):

        _names = ['const'] + self.param_names
        _valid_result = self.bootstrapResult[_names].dropna()

        coef = _valid_result.median() 
        se = _valid_result.std() 
        mad = _valid_result.apply(lambda x:np.median(np.abs(x - np.median(x))))
        ci_lower = _valid_result.apply(lambda x:sorted(x)[int(sig/2 * len(_valid_result))])
        ci_upper = _valid_result.apply(lambda x:sorted(x,reverse=True)[int(sig/2 * len(_valid_result))])
        
        return pd.DataFrame({'median_coef':coef,'se':se,'mad':mad,'ci_lower':ci_lower,'ci_upper':ci_upper})
    
   
    def get_cond_loss(self,result,b=2):

        y = self.data['value_surplus']
        X = sm.add_constant(self.data[self.endog_cols]).astype(float)

        _sresid = (y - result.predict(X)) / self.model.scale

        loss = _sresid.apply(lambda x:min(x**2,b**2)) @ self.model.weights

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
    

    # def subject_cluster_se(self):
        
    #     all_workers = self.data['worker_id'].unique()

    #     # Initialize W as a zero matrix with shape (n, n) where n is the number of observations
    #     W = np.zeros((int(self.model.nobs), int(self.model.nobs)))

    #     # Assign 1 to diagonal elements corresponding to observations within the same cluster
    #     for cluster_id in all_workers:
    #         _indices = self.data[self.data['worker_id'] == cluster_id].index
    #         W[_indices, _indices] = 1

    #     # "bread" term of the sandwich estimator
    #     X = sm.add_constant(self.data[self.param_names]).astype(float)
    #     inv_X_squared = np.linalg.inv(np.dot(X.T, X))

    #     # "meat" term of the sandwich estimator
    #     resid_squared = np.dot(self.model.resid, self.model.resid.T)
    #     Omega = np.dot(resid_squared,W)

    #     # clustered covariance
    #     clustered_cov = inv_X_squared @ np.dot(np.dot(X.T,Omega),X) @ inv_X_squared
        
    #     return np.sqrt(np.diag(clustered_cov))
    
    # def subject_cluster_pvalues(self):
        
    #     _names = ['const'] + self.param_names

    #     coef = self.model.params[_names]
    #     se = self.subject_cluster_se()
    #     score = np.abs(coef / se)

    #     p_value = 2*(1 - stats.t.cdf(score, self.model.nobs - len(_names) - 1))

    #     return p_value


def get_star(p):
    if p > 0.05:
        return ''
    elif p > 0.01:
        return '$^{*}$'
    elif p > 0.005:
        return '$^{**}$'
    else:
        return '$^{***}$'


def draw_df_from_result(reg_result,col_names,digit=3):

    result_table = {k:[] for k in col_names}
    result_table['model'] = list(reg_result.keys())

    for r in col_names:
        for m in list(reg_result.keys()):
            b_coef = r.split('b_',1)
            se_coef = r.split('se_',1)
            coef_name = reg_result[m]['coef_name']

            if len(b_coef) > 1 and b_coef[1] in coef_name:
                _b = reg_result[m]['params'][coef_name.index(b_coef[1])]
                _p_value = reg_result[m]['pvalues'][coef_name.index(b_coef[1])]

                if b_coef[1] == 'const':
                    _b = _b + reg_result[m]['contrast_mean']

                result_table[r] += [ str(round(_b,digit)) + get_star(_p_value)]

            elif len(se_coef) > 1 and se_coef[1] in coef_name:
                _se = reg_result[m]['bse'][coef_name.index(se_coef[1])]
                result_table[r] += [ '(' + str(round(_se,digit)) + ')' ]

            elif r == 'nobs':
                result_table[r] += [ int(reg_result[m][r]) ]
            
            elif r == 'rsquared_adj':
                try:
                    result_table[r] += [ str(round(reg_result[m][r],digit)) ]
                except:
                    result_table[r] += ['']
                
            else:
                result_table[r] += ['']

    return pd.DataFrame(result_table)




def add_border(input_string):

    # Replace '\toprule', '\midrule', '\bottomrule' with '\hline'
    output_string = input_string.replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule', '\\hline')
    
    # Insert '\hline' before '\nobservations'
    index = output_string.find('\nobservations')
    output_string = output_string[:index] + '\\hline\n' + output_string[index:]

    return output_string


def make_table(input_df,output_path):
    with open(output_path,'w') as f:
        # tex_code = '\\documentclass[12px]{article} \n \\begin{document} \n' + input_df.to_latex() + '\n \end{document}'
        tex_code = input_df.to_latex()
        tex_code = add_border(tex_code)
        f.write(tex_code)

    

def get_ci(model,digit=3):

    row_names_ci = ['front_amount_6m',
             'front_amount_12m',
             'front_amount_6m_0',
             'front_amount_12m_0',
             'front_amount_6m_1',
             'front_amount_12m_1',
             'choice_peli',
             'const']
    
    ci_table = model.conf_int()
    ci_list = []
    for r in row_names_ci:
        if r in ci_table.index:
            _lower = np.round(ci_table.loc[r]['ci_lower'],digit)
            _upper = np.round(ci_table.loc[r]['ci_upper'],digit)
            ci_list += ['[' + str(_lower) + ', ' + str(_upper) + ']']
        else:
            ci_list += ['']

    return ci_list
    


