
import numpy as np
from mpl import choice_rule
from mpl import estimation
from tqdm import tqdm


def split_sample(data,train_size):
    
    subject_id_set = set(data["person_id"])

    train_size = round(len(subject_id_set)* train_size)
    train_subject_id = np.random.choice(np.array(list(subject_id_set)),size=train_size,replace=False)
    
    train_sample = data[data["person_id"].isin(train_subject_id)]
    test_sample = data[data["person_id"].isin(train_subject_id) == False]

    return {"train": train_sample,"test":test_sample}



def test_model(style,test_sample,params):

    ss_t = test_sample['ss_t'].values
    ss_x = test_sample['ss_x'].values
    ll_t = test_sample['ll_t'].values
    ll_x = test_sample['ll_x'].values
    choice = test_sample['choice'].values


    predict_choice = choice_rule.choice_prob(ss_x, ss_t, ll_x, ll_t, 
                            dstyle = style["dstyle"], 
                            ustyle = style["ustyle"], 
                            method = style['method'],
                            params = params[:-1], 
                            temper = params[-1])

    choice_not_nan = (~np.isnan((choice - predict_choice)))

    choice_valid = choice[choice_not_nan]
    predict_valid = predict_choice[choice_not_nan]
    resid = choice_valid - predict_valid

    mse = np.var(resid)
    mae = abs(resid).mean()
    accuracy = (abs(resid)<.5).sum()/len(resid)
    p_choice = np.where(choice_valid == 1, predict_valid, 1 - predict_valid)
    log_like = np.sum(np.log(p_choice))

    return {"mse":mse,"mae":mae,"accuracy":accuracy,"log_like":log_like}



def validation(style,data,sample_times,train_size=0.75,disp_output=False):

    test_result = []

    for i in tqdm(range(sample_times)):

        sample = split_sample(data,train_size)

        train_result = mpl_estimation.mle(style,data = sample["train"])

        if train_result == "Fail to converge":
            test_model_result = {"test_result": "Fail to converge"}
        else:
            test_model_result = test_model(style=style,test_sample=sample["test"],params=train_result["params"])
        
        if disp_output:
            print(test_model_result)    

        test_model_result["iter"] = i
        test_model_result["model"] = style["dstyle"] + "-" + style["ustyle"]

        test_result += [test_model_result]

    
    return test_result