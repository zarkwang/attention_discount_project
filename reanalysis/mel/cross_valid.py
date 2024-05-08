
import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
from mel import choice_rule
from mel import estimation
from tqdm import tqdm
from sklearn import metrics
from sklearn import model_selection


class data_prepare:

    def __init__(self,data,
                 feature=None,label=None,group=None):

        self._data = data
        self._feature = feature
        self._label = label
        self._group = group

    def generate_features(self):

        dt = self._data.dropna()

        pd.set_option('mode.chained_assignment', None)

        dt['ratio_x'] = dt['ll_x'] / dt['ss_x']
        dt['abs_diff_x'] = dt['ll_x'] - dt['ss_x']
        dt['abs_diff_t'] = dt['ll_t'] - dt['ss_t']
        dt['rel_diff_x'] = 2*(dt['ll_x'] - dt['ss_x'])/(dt['ll_x'] + dt['ss_x']) 
        dt['rel_diff_t'] = 2*(dt['ll_t'] - dt['ss_t'])/(dt['ll_t'] + dt['ss_t']) 
        dt['growth_x'] = np.log(dt['ratio_x']) / dt['abs_diff_t']

        self._data = dt

    def split_sample(self,test_size=0.2,split_X_y=True):
        
        groups = self._data[self._group]
        
        test_index = np.random.choice(list(set(groups)),size=int(len(set(groups)) * test_size),replace=False)
        train_index = np.array(list(set(groups)-set(test_index)))
        
        self.train_sample = self._data[self._data[self._group].isin(train_index)]
        self.test_sample = self._data[self._data[self._group].isin(test_index)]

        if split_X_y:
            X_train = self.train_sample[self._feature]
            X_test = self.test_sample[self._feature]
            y_train = self.train_sample[self._label]
            y_test = self.test_sample[self._label]
        
        return X_train,X_test,y_train,y_test




def test_model(test_sample=None,label='choice',
               y_test=None,X_test=None,
               style=None,params=None,model=None,output='scores'):
    
    if X_test is None:
        y_test = test_sample[label]
        X_test = test_sample.drop(label,axis=1)

    if style == 'heuristic':
        
        preds = np.array([x[1] for x in model.predict_proba(X_test)])
        pred_binary = (preds > .5)

        if output == 'predict_proba':
            return preds
        elif output == 'scores':
            
            scores = {'dstyle': 'heurstic',
                    'ustyle': '--',
                    'mse': metrics.mean_squared_error(y_test, preds),
                    'mae': metrics.mean_absolute_error(y_test, preds),
                    'log_loss': metrics.log_loss(y_test, preds),
                    'accuracy': metrics.accuracy_score(y_test,pred_binary),
                    'pred_ll':sum(pred_binary)/len(pred_binary)
                }
            
            return scores
        else:
            print("output can only be 'predict_proba' or 'scores'")

    else:
        ss_t = X_test['ss_t'].values
        ss_x = X_test['ss_x'].values
        ll_t = X_test['ll_t'].values
        ll_x = X_test['ll_x'].values

        preds = choice_rule.choice_prob(ss_x, ss_t, ll_x, ll_t, 
                                dstyle = style['dstyle'], 
                                ustyle = style['ustyle'], 
                                method = style['method'],
                                intercept=style['intercept'],
                                params = params[:-1], 
                                temper = params[-1])
        
        if output == 'predict_proba':
            return preds
        elif output == 'scores':
            choice = y_test.values
            choice_not_nan = (~np.isnan((choice - preds)))

            choice_valid = choice[choice_not_nan]
            predict_valid = preds[choice_not_nan]
            pred_binary = (predict_valid > .5)

            scores= {"mse": metrics.mean_squared_error(choice_valid,predict_valid),
                        "mae": metrics.mean_absolute_error(choice_valid,predict_valid),
                        "log_loss": metrics.log_loss(choice_valid,predict_valid),
                        "accuracy": metrics.accuracy_score(choice_valid,pred_binary),
                        "pred_ll": sum(pred_binary)/len(predict_valid)}
            return scores
        else:
            print("output can only be 'predict_proba' or 'scores'")

        
    


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
                                groups=data[kwargs['groups']]))
        else:
            self._cv = cv
        
        
        n_style = len(self._style)
        n_cv = len(self._cv)
        self.fit_list = list(itertools.product(np.arange(n_style), np.arange(n_cv)))
        self.n_fit = len(self.fit_list)

    
    def fit(self):
        
        if self.n_jobs==0:
            self.val_styles,self.val_params,self.val_scores,self.success = self.KFvalidation_SingleProcess()
        else:
            self.val_styles,self.val_params,self.val_scores,self.success = self.KFvalidation_MultiProcess(self.n_jobs)
        


    @staticmethod
    def train_val_func(style,train_set,val_set,n_max_attempt=10):

        train_result = estimation.config_param['msg']['fail_converge']
        x0 = None

        for i in range(n_max_attempt):
            mle = estimation.mle(style=style,data=train_set,x0=x0)
            output = mle.solve()
            
            if output == estimation.config_param['msg']['fail_converge']:
                if isinstance(mle.params,np.ndarray):
                    x0 = mle.params
                else:
                    x0 = None
                print(f'Attempt times: {i+1}')
                print(output)
                continue 
            else:
                train_result = output 
                break
        
        params = mle.params
        val_result = test_model(style=style,test_sample=val_set,params=mle.params)
        val_scores = list(val_result.values())

        success = (train_result != estimation.config_param['msg']['fail_converge'])
            
        return params,val_scores,success
    

    def fit_func(self,i):

        style_id = self.fit_list[i][0]
        cv_id = self.fit_list[i][1]

        input_style = self._style[style_id]

        train_set = self._data[self._data.index.isin(self._cv[cv_id][0])]
        val_set = self._data[self._data.index.isin(self._cv[cv_id][1])]

        param,score,success = self.train_val_func(input_style,train_set,val_set,
                                                        n_max_attempt=self.n_max_attempt)

        return input_style,param,score,success

    
    def KFvalidation_SingleProcess(self):
        
        val_styles = []
        val_params = []
        val_scores = []
        val_success = []

        for i in tqdm(self.fit_list):

            input_style,param,score,success = self.fit_func(i)

            val_styles.append(input_style)
            val_params.append(param)
            val_scores.append(score)
            val_success.append(success)

        return val_styles,val_params,val_scores,val_success



    def KFvalidation_MultiProcess(self,n_jobs):

        with mp.Pool(processes=n_jobs) as pool:
            
            val_styles = []
            val_params = []
            val_scores = []
            val_success = []
            
            for input_style,param,score,success in tqdm(
                                                    pool.imap(self.fit_func,range(self.n_fit)),
                                                    total=self.n_fit):
                
                val_styles.append(input_style)
                val_params.append(param)
                val_scores.append(score)
                val_success.append(success)

        pool.join()
        pool.close()

        return val_styles,val_params,val_scores,val_success
    

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
        sum_tab['dstyle'] = sum_tab['style'].apply(lambda x:x['dstyle'])
        sum_tab['ustyle'] = sum_tab['style'].apply(lambda x:x['ustyle'])

        df_cols = sum_tab.columns.tolist()
        df_cols = df_cols[-2:] + df_cols[:-2]
        sum_tab = sum_tab.reindex(columns=df_cols)

        return sum_tab


   


def get_result_tab(kf_result_df,test_sample):

    test_result = []

    for i in range(len(kf_result_df)):

        test_style = kf_result_df['style'][i]
        test_params = kf_result_df['params'][i]

        test_scores = test_model(test_sample=test_sample,style=test_style,params=test_params)
        test_scores['dstyle'] = test_style['dstyle'] 
        test_scores['ustyle'] = test_style['ustyle']
        test_result.append(test_scores)

    test_result = pd.DataFrame(test_result)
    test_result = test_result.reindex(columns=kf_result_df.columns).sort_values('accuracy',ascending=False)
    
    return test_result.drop(['style','params'],axis=1)