
import os
import json
import pandas as pd
from datetime import datetime

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


if __name__ == '__main__':

    
    folder_path = os.path.join(os.getcwd(), 'jatos_results_20240125')
    metadata_path = os.path.join(folder_path, 'metadata.json')

    # create meta dataset: prolific_id, duration in each component, age, sex  
    with open(metadata_path, 'r') as file:
                metadata = json.load(file)

    study_result = metadata['data'][0]['studyResults']

    df_meta = read_meta_file(study_result)
    
    df_demographic = pd.read_csv('prolific_demographic.csv')[['Participant id','Age','Sex']].\
                                rename(columns={'Participant id':'prolific_id',
                                                'Age':'age','Sex':'sex'})

    df_meta = pd.merge(df_meta,df_demographic,on='prolific_id')

    print("Meta data is created.")

    
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

    with pd.ExcelWriter('all_components_data.xlsx') as writer:
        df_time.to_excel(writer, sheet_name='fill_in_blank', index=False)
        df_consistency.to_excel(writer, sheet_name='consistency_check', index=False)
        df_peli.to_excel(writer, sheet_name='peli', index=False)
        df_meta.to_excel(writer,sheet_name='meta_data',index=False)

    print("Main data is loaded successfully.")
