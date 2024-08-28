import pandas as pd
from sklearn.cluster import KMeans

print('--data_loading')
df = pd.read_csv('preprocessing_data/5_test_community_net_info.csv')

# NaN 값을 특정 값으로 대체 (예: -1)
df = df.fillna(-100)

print('--k-means')
# K-means 알고리즘을 사용하여 클러스터링
num_clusters = 6
features = df.drop('Community', axis=1)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(features)

print('--result saving')
result = pd.DataFrame({'community':df['Community'],'Cluster':df['cluster']})
# 결과 저장
result.to_csv('preprocessing_data/6_test_community_clustering.csv', index=False)


# 각 클러스터별로 컬럼값의 평균 계산
cluster_means = df.groupby('cluster').mean()

# 결과 출력
print("각 클러스터별 컬럼값의 평균:")
print(cluster_means)

