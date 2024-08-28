import pandas as pd

# # 첫 번째 데이터 프레임 생성
# df1 = pd.read_csv('preprocessing_data/4_test_node_community_info.csv')

# # 두 번째 데이터 프레임 생성
# df2 = pd.read_csv('preprocessing_data/6_test_community_clustering.csv')

# df3 = pd.read_csv('preprocessing_data/1_test_total_amount_spent.csv')

df4 = pd.read_csv('source_data/network_info.csv')
df = df4[['Id', 'amount_spent', 'modularity_class']]

df2 = pd.read_csv('clustered_data_result.csv')
df3 = pd.read_csv('test2.csv')
# 두 데이터 프레임을 community 칼럼을 기준으로 조인
merged_df = pd.merge(df, df2, on='modularity_class', how='left')
merged_df = pd.merge(merged_df, df3, on='Id', how='left')
# merged_df = pd.merge(merged_df, df4, on='id', how='left')


# 결과 저장
merged_df.to_csv('preprocessing_data/7_id_info_join.csv', index=False)