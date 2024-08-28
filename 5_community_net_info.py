import pandas as pd
import networkx as nx
from statistics import mean, variance

# CSV 파일을 읽어온다
print('---trade data loading')
df = pd.read_csv('preprocessing_data/3_test_no_pay_trade.csv')


# 그래프 생성
print('---graph generating')
G = nx.from_pandas_edgelist(df, 'source', 'target')

# 커뮤니티 정보가 있는 CSV 파일을 읽어온다
print('---node community info data loading')
community_df = pd.read_csv('preprocessing_data/4_test_node_community_info.csv')

# 각 노드에 속한 커뮤니티 정보를 추가
print('---community info adding to node')
for index, row in community_df.iterrows():
    node_id = row['id']
    community_id = row['community']
    G.nodes[node_id]['community'] = community_id

# 각 커뮤니티의 degree centrality, betweenness centrality, assortativity, clustering coefficient, radius 계산
degree_centrality_avg = []
degree_centrality_var = []
betweenness_centrality_avg = []
betweenness_centrality_var = []
degree_assortativity = []
clustering_coefficient_avg = []
radius = []
community_size = []

print('---community-net-info calculating')
for community_id, community_nodes in community_df.groupby('community')['id']:
    print('>>community name : ',community_id,' calc...')
    subgraph = G.subgraph(community_nodes)
    print(' community_size : ', len(community_nodes))
    community_size.append(len(community_nodes))
    degree_centrality = nx.degree_centrality(subgraph)
    print(' degree_centrality : ', mean(degree_centrality.values()))
    betweenness_centrality = nx.betweenness_centrality(subgraph)
    print(' betweenness_centrality : ', mean(betweenness_centrality.values()))
    degree_centrality_avg.append(mean(degree_centrality.values()))
    degree_centrality_var.append(variance(degree_centrality.values()))
    betweenness_centrality_avg.append(mean(betweenness_centrality.values()))
    betweenness_centrality_var.append(variance(betweenness_centrality.values()))
    
    # Assortativity 계산
    degree_assortativity.append(nx.degree_assortativity_coefficient(subgraph))
    print(' degree_assortativity : ', degree_assortativity)
    
    # Clustering Coefficient 계산
    clustering_coefficient_avg.append(nx.average_clustering(subgraph))
    
    # Radius 계산
    radius.append(nx.radius(subgraph))
    

print('---saving data')
# 결과를 DataFrame으로 저장
result_df = pd.DataFrame({
    'Community': community_df['community'].unique(),
    'DegreeCentralityAvg': degree_centrality_avg,
    'DegreeCentralityVar': degree_centrality_var,
    'BetweennessCentralityAvg': betweenness_centrality_avg,
    'BetweennessCentralityVar': betweenness_centrality_var,
    'DegreeAssortativity': degree_assortativity,
    'ClusteringCoefficientAvg': clustering_coefficient_avg,
    'Radius': radius,
    'CommunitySize': community_size
})

result_df.to_csv('preprocessing_data/5_test_community_net_info.csv', index=False)
