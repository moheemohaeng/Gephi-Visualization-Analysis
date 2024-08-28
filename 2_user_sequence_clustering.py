import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset

##### 유저별 활동 내역 시퀀스 클러스터링 #####
print('---data loading')
df = pd.read_csv('source_data/test1_activity.csv')

# 날짜를 datetime 형식으로 변환
df['day'] = pd.to_datetime(df['day'])


# 각 유저의 시퀀스 데이터 생성
print('---create sequence data')
sequences = df.groupby('acc_id')[['playtime','npc_kill','solo_exp','party_exp','quest_exp','rich_monster','death','revive','exp_recovery','fishing','private_shop','game_money_change','enchant_count'
]].apply(lambda x: x.values.tolist()).reset_index(name='sequence')


# 시퀀스 데이터의 길이를 맞추기 위해 패딩
print('---sequence data padding')
max_length = sequences['sequence'].apply(len).max()
sequences['sequence'] = sequences['sequence'].apply(lambda x: x + [[0]*13] * (max_length - len(x)))
# 수치형 데이터만 추출하여 3D 배열로 변환
X_numeric = np.array(sequences['sequence'].tolist())


# 시계열 데이터를 정규화
# print('---data normalization')
# X_numeric = TimeSeriesScalerMinMax().fit_transform(X_numeric)

# K-means 클러스터링
num_clusters = 5
batch_size = 2  # 나눠서 처리할 배치 크기 설정

# 결과를 저장할 데이터프레임 생성
result_df = pd.DataFrame(columns=['acc_id', 'cluster_label'])

# print('---clustering')
# # 데이터를 배치로 나누어 정규화 및 클러스터링 수행
# for i in range(0, len(X_numeric), batch_size):
#     batch_X = X_numeric[i:i+batch_size]
    
#     # 시계열 데이터를 정규화
#     scaler = TimeSeriesScalerMeanVariance()
#     batch_X_normalized = scaler.fit_transform(batch_X)
    
#     kmeans = TimeSeriesKMeans(n_clusters=num_clusters, verbose=True, random_state=42)
#     labels = kmeans.fit_predict(batch_X_normalized)
    
#     # 각 acc_id에 대한 클러스터 레이블을 추가
#     batch_result_df = pd.DataFrame({'acc_id': sequences['acc_id'][i:i+batch_size], 'cluster_label': labels})
    
#     # 결과를 저장
#     result_df = pd.concat([result_df, batch_result_df], ignore_index=True)

# print('---save data')
# # 결과 출력
# result_df.to_csv('preprocessing_data/2_user_sequence_clustering.csv')

# exit()






# 시계열 데이터를 tslearn의 형식에 맞게 변환
print('---transform data')
X = to_time_series_dataset([X_numeric[i] for i in range(X_numeric.shape[0])])

# 시계열 데이터를 정규화
print('---data normalization')
X = TimeSeriesScalerMinMax().fit_transform(X)

# K-means 클러스터링
print('---k-means clustering...')
num_clusters = 5
kmeans = TimeSeriesKMeans(n_clusters=num_clusters, verbose=True, random_state=42)
labels = kmeans.fit_predict(X)

# 각 acc_id에 대한 클러스터 레이블을 추가
print('---label adding...')
sequences['cluster_label'] = labels

# acc_id와 클러스터 레이블만을 따로 저장
print('---save...')
result_df = sequences[['acc_id', 'cluster_label']].set_index('acc_id')

# 결과 저장
result_df.to_csv('preprocessing_data/2_user_sequence_clustering_test.csv')