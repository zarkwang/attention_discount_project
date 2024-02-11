
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

pd.options.mode.chained_assignment = None


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

    folder_path = os.getcwd()
    out_main_data_path = 'valid_sequence_data.csv'

    # load the datasets
    df_consistency = pd.read_excel('all_components_data.xlsx',sheet_name = 'consistency_check')
    df_time = pd.read_excel('all_components_data.xlsx',sheet_name = 'fill_in_blank')
    df_peli = pd.read_excel('all_components_data.xlsx',sheet_name = 'peli')

    # consistency check
    df_check = pd.merge(df_consistency,df_time,on=['worker_id','front_amount','seq_length'])

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

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(df_time_pivot.values)
    gmm_labels = gmm.predict(df_time_pivot.values)

    # cluster results
    df_time_pivot['label_kmeans'] = kmeans.labels_
    df_time_pivot['label_gmm'] = np.abs(gmm_labels - 1)
    df_time_pivot.to_csv('cluster_result_data.csv')

    cols = ['label_kmeans','label_gmm'] + [col for col in df_time_pivot if (col != 'label_kmeans') and (col != 'label_gmm')]
    df_time_pivot = df_time_pivot[cols].reset_index()
    print('Number of subjects in each cluster:')
    print('K-means',np.bincount(df_time_pivot['label_kmeans']))
    print('GMM',np.bincount(df_time_pivot['label_gmm']))

    # save the data
    df_time_label = pd.merge(df_time_valid,df_time_pivot[['worker_id','label_kmeans','label_gmm']],on=['worker_id'])
    df_time_label = df_time_label.rename(columns={'choice_value':'choice_peli'})
    df_time_label.to_csv('valid_sequence_data.csv')