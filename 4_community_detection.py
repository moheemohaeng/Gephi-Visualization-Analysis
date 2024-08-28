import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

# CSV 파일을 읽어온다
# df = pd.read_csv('source_data/3_no_pay_trade_test.csv')
df = pd.read_csv('preprocessing_data/3_no_pay_trade_test.csv')

# 그래프 생성
G = nx.from_pandas_edgelist(df, 'source_acc_id', 'target_acc_id')

# Greedy 모듈러리티 커뮤니티 탐지
print('---community detection start')
# communities = list(greedy_modularity_communities(G))
# communities = list(nx.algorithms.community.girvan_newman(G))
# communities = list(nx.algorithms.community.label_propagation.label_propagation_communities(G))
communities = list(nx.algorithms.community.greedy_modularity_communities(G))

# 각 노드에 속한 커뮤니티 정보를 추가
print('---community info add')
for idx, community in enumerate(communities):
    for node in community:
        G.nodes[node]['community'] = idx

# 각 노드의 커뮤니티 정보를 노드의 속성으로 저장
print('--saving')
community_info = nx.get_node_attributes(G, 'community')

# 노드 ID와 커뮤니티 정보를 데이터프레임으로 저장
df_community_info = pd.DataFrame(list(community_info.items()), columns=['id', 'community'])

# CSV 파일로 저장
df_community_info.to_csv('preprocessing_data/4_test_node_community_info.csv', index=False)