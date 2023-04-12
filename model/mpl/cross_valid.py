
import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
from mpl import choice_rule
from mpl import estimation
from tqdm import tqdm
from sklearn import metrics
from sklearn import model_selection
from functools import partial


def generate_sample(data):

    dt = data.dropna()

    pd.set_option('mode.chained_assignment', None)

    dt['ratio_x'] = dt['ll_x'] / dt['ss_x']
    dt['abs_diff_x'] = dt['ll_x'] - dt['ss_x']
    dt['abs_diff_t'] = dt['ll_t'] - dt['ss_t']
    dt['rel_diff_x'] = 2*(dt['ll_x'] - dt['ss_x'])/(dt['ll_x'] + dt['ss_x']) 
    dt['rel_diff_t'] = 2*(dt['ll_t'] - dt['ss_t'])/(dt['ll_t'] + dt['ss_t']) 
    dt['growth_x'] = np.log(dt['ratio_x']) / dt['abs_diff_t']

    return dt



def test_model(style,test_sample,params):

    ss_t = test_sample['ss_t'].values
    ss_x = test_sample['ss_x'].values
    ll_t = test_sample['ll_t'].values
    ll_x = test_sample['ll_x'].values
    choice = test_sample['choice'].values


    predict_choice = choice_rule.choice_prob(ss_x, ss_t, ll_x, ll_t, 
                            dstyle = style['dstyle'], 
                            ustyle = style['ustyle'], 
                            method = style['method'],
                            intercept=style['intercept'],
                            params = params[:-1], 
                            temper = params[-1])

    choice_not_nan = (~np.isnan((choice - predict_choice)))

    choice_valid = choice[choice_not_nan]
    predict_valid = predict_choice[choice_not_nan]
    binary_pred = (predict_valid > .5)

    mse = metrics.mean_squared_error(choice_valid,predict_valid)
    mae = metrics.mean_absolute_error(choice_valid,predict_valid)
    log_loss =metrics.log_loss(choice_valid,predict_valid)
    accuracy = metrics.accuracy_score(choice_valid,binary_pred)

    test_scores= {"mse":mse,
                "mae":mae,
                "log_loss":log_loss,
                "accuracy":accuracy,
                "pred_ll":sum(binary_pred)/len(predict_valid)}
    
    return test_scores




class KFvalidation:

    def __init__(self,style,data,
                 cv=None,random_state=2023,n_jobs=0,n_max_attempt=10,
                 **kwargs):
        
        self._data = data
        self.n_max_attempt = n_max_attempt
        self.n_jobs = n_jobs

        if not isinstance(style,list):
            self._style = [style]
        else:
            self._style = style
        
        if not cv:
            sgkf = model_selection.StratifiedGroupKFold(n_splits=kwargs['n_splits'],
                                                        shuffle=True,
                                                        random_state=random_state)
            
            self._cv = list(sgkf.split(X=data[kwargs['features']],
                                y=data[kwargs['label']],
                                groups=data[kwargs['groups']]))
        else:
            self._cv = cv
        
        
        n_style = len(self._style)
        n_cv = len(self._cv)

        self.n_fit = n_cv*n_style
        self.fit_list = list(itertools.product(np.arange(n_style), np.arange(n_cv)))

    
    def fit(self):
        
        if self.n_jobs==0:
            self.val_styles,self.val_params,self.val_scores = self.KFvalidation_SingleProcess()
        else:
            self.val_styles,self.val_params,self.val_scores = self.KFvalidation_MultiProcess(self.n_jobs)
        
        self.fails = (np.array(self.val_params,dtype=object) == estimation.config_param['msg']['fail_converge'])


    @staticmethod
    def train_val_func(style,train_set,val_set,n_max_attempt=10):

        train_result = estimation.config_param['msg']['fail_converge']

        for i in range(n_max_attempt):
            output = estimation.mle(style=style,data=train_set)
            
            if output == estimation.config_param['msg']['fail_converge']:
                print(estimation.config_param['msg']['fail_converge'],f'Attempt times: {i}.')
                continue 
            else:
                train_result = output 
                break

       
        val_result = test_model(style=style,test_sample=val_set,params=train_result["params"])
        s = style
        p = [round(e,3) for e in train_result['params']]
        v = [round(e,3) for e in list(val_result.values())]

        return s,p,v
    

    def fit_func(self,i):

        style_id = self.fit_list[i][0]
        cv_id = self.fit_list[i][1]

        input_style = self._style[style_id]

        train_set = self._data[self._data.index.isin(self._cv[cv_id][0])]
        val_set = self._data[self._data.index.isin(self._cv[cv_id][1])]

        s,p,v = self.train_val_func(input_style,train_set,val_set,n_max_attempt=self.n_max_attempt)

        return s,p,v

    
    def KFvalidation_SingleProcess(self):
        
        val_styles = []
        val_params = []
        val_scores = []

        for i in tqdm(self.fit_list):

            s,p,v = self.fit_func(i)

            val_styles.append(s)
            val_params.append(p)
            val_scores.append(v)

        return val_styles,val_params,val_scores



    def KFvalidation_MultiProcess(self,n_jobs):

        with mp.Pool(processes=n_jobs) as pool:
            
            val_styles = []
            val_params = []
            val_scores = []
            
            for s,p,v in tqdm(pool.imap(self.fit_func,range(self.n_fit)),total=self.n_fit):
                
                val_styles.append(s)
                val_params.append(p)
                val_scores.append(v)

        pool.join()
        pool.close()

        return val_styles,val_params,val_scores
    

    def summary(self):

        df = pd.DataFrame({'style':self.val_styles,
                            'params':self.val_params,
                            'scores':self.val_scores})

        df['style'] = df['style'].astype(str)

        score_list = ['mse', 'mae', 'log_loss','accuracy','pred_ll']
        df[score_list] = df.scores.apply(lambda x:pd.Series(x))

        df_group = df.groupby('style')
        sum_tab = df_group.params.apply(lambda x: np.around(np.mean(x.tolist(),axis=0),decimals=3)).to_frame()
        sum_tab[score_list] = df_group[score_list].mean()

        sum_tab = sum_tab.reset_index()
        sum_tab['style'] = sum_tab['style'].apply(lambda x:eval(x))

        return sum_tab


   



    