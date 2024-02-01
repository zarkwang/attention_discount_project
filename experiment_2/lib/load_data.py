
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans

pd.options.mode.chained_assignment = None


def read_meta_file(study_result):

    comp_cols = ['component_duration_' + str(i+1) for i in range(len(study_result[0]['componentResults']))]
    meta_cols = ['worker_id','prolific_id','status','duration','duration_second'] + comp_cols

    dict_meta = {key: [] for key in meta_cols}

    for w in range(len(study_result)):
        if study_result[w]['studyState'] == 'FINISHED':
            dict_meta['worker_id'] += [str(study_result[w]['workerId'])]
            dict_meta['prolific_id'] += [study_result[w]['urlQueryParameters']['PROLIFIC_PID']]
            dict_meta['status'] += [study_result[w]['studyState']]
            dict_meta['duration'] += [study_result[w]['duration']]
            time_object = datetime.strptime(study_result[w]['duration'], "%H:%M:%S")
            duration_second = time_object.hour * 3600 + time_object.minute * 60 + time_object.second
            dict_meta['duration_second'] += [duration_second]
            for i in range(len(comp_cols)):
                dict_meta[comp_cols[i]] += [study_result[w]['componentResults'][i]['duration']]

    df_meta = pd.DataFrame(dict_meta)

    return df_meta


def read_comp_result(file_path,df_consistency,df_time,df_peli):

    with open(file_path, 'r') as file:
            data = json.load(file)

    if 'task' in data:
        if data['task'] == 'consistency-check':
            df_consistency_new = pd.DataFrame(data['choice'])
            df_consistency_new['worker_id'] = data['worker_id']
            df_consistency_new['screen_width'] = data['screen_width']
            df_consistency = pd.concat([df_consistency, df_consistency_new], ignore_index=True)
        elif data['task'] == 'blank-filling':
            df_time_new = pd.DataFrame(data['choice'])
            df_time_new['worker_id'] = data['worker_id']
            df_time_new['prolific_id'] = data['url']['PROLIFIC_PID']
            df_time = pd.concat([df_time, df_time_new], ignore_index=True)
        elif data['task'] == 'income-earlier-later':
            url = data.pop('url')
            df_peli_new = pd.DataFrame(data,index=[data['worker_id']])
            df_peli_new['prolific_id'] = url['PROLIFIC_PID']
            df_peli = pd.concat([df_peli, df_peli_new], ignore_index=True)

    return df_consistency,df_time,df_peli


def check_fail(row):
    '''
    If choosing two-reward sequence:
      indifference point should not be smaller than the single-reward sequence
    If choosing single reward: 
      indifference point should not be larger than the single reward
    '''

    if row['sequence_single_choice'] == 'sequence':
        return row['indiff_point'] < row['single_amount']
    elif row['sequence_single_choice'] == 'single':
        return row['indiff_point'] > row['single_amount']
    else:
        return True


if __name__ == '__main__':

    
    folder_path = os.path.join(os.getcwd(), 'jatos_results_20240125')
    metadata_path = os.path.join(folder_path, 'metadata.json')
    out_meta_path = 'meta_data.csv'
    out_main_data_path = 'valid_sequence_data.csv'

    # create meta dataset: prolific_id, duration in each component, age, sex  
    with open(metadata_path, 'r') as file:
                metadata = json.load(file)

    study_result = metadata['data'][0]['studyResults']

    df_meta = read_meta_file(study_result)
    
    df_demographic = pd.read_csv('prolific_demographic.csv')[['Participant id','Age','Sex']].\
                                rename(columns={'Participant id':'prolific_id',
                                                'Age':'age','Sex':'sex'})

    df_meta = pd.merge(df_meta,df_demographic,on='prolific_id')
    df_meta.to_csv(out_meta_path)

    print("Save meta data as", out_meta_path)

    # create main datasets
    df_consistency = pd.DataFrame() # consistency check questions
    df_time = pd.DataFrame() # fill-in-the-blank questions
    df_peli = pd.DataFrame() # preference for earlier vs later income (PELI) task

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("data.txt"):
                file_path = os.path.join(root, file)
                df_consistency,df_time,df_peli = read_comp_result(file_path,df_consistency,df_time,df_peli)

    df_consistency.insert(0, 'worker_id', df_consistency.pop('worker_id'))
    df_time.insert(0, 'worker_id', df_time.pop('worker_id'))
    df_peli.insert(0, 'worker_id', df_peli.pop('worker_id'))

    print("Main data loaded successfully.")

    # consistency check
    df_check = pd.merge(df_consistency,df_time,on=['worker_id','front_amount','backend_amount'])

    df_check['fail_check'] = df_check.apply(check_fail, axis=1)

    df_check_result = df_check.groupby('worker_id')['fail_check'].sum() > 0
    fail_consistency_check = df_check_result[df_check_result].index.unique()

    print('Number of participants who fail consistency check:',len(fail_consistency_check))
    print('Worker IDs to be excluded:',list(fail_consistency_check))

    # merge all answers into one dataset
    df_time_valid = df_time[~df_time['worker_id'].isin(fail_consistency_check)]
    df_time_valid['value_surplus'] = df_time_valid['indiff_point'] - df_time_valid['front_amount']
    df_time_valid = pd.merge(df_time_valid,
                            df_peli[['worker_id','choice_label','choice_value','response_time_mel']],on='worker_id'). \
                        sort_values(['worker_id','q_id'])
    
    print("Consistency check finished.")

    # cluster analysis: divide data into two groups using k-means clustering
    df_time_valid['reward'] = df_time_valid.apply(lambda x: 'reward_' + x['seq_length'].split()[0] +'_' + str(x['front_amount']),axis=1)
    df_time_pivot = df_time_valid.pivot(index='worker_id', columns='reward', values='value_surplus')

    kmeans = KMeans(n_clusters=2,random_state=42)
    kmeans.fit(df_time_pivot.values)

    # cluster results
    df_time_pivot['label'] = kmeans.labels_
    cols = ['label'] + [col for col in df_time_pivot if col != 'label']
    df_time_pivot = df_time_pivot[cols].reset_index()
    print('Number of participants in each cluster:',np.bincount(kmeans.labels_))

    # save the data
    df_time_label = pd.merge(df_time_valid,df_time_pivot[['worker_id','label']],on=['worker_id'])
    df_time_label.to_csv('valid_sequence_data.csv')