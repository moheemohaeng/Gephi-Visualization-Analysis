import pandas as pd

##### 대가 없이 교환된 거래 정보만 추출 ##### clear

# train.csv 파일을 읽어옴
df = pd.read_csv('source_data/test1_trade.csv')

# 'item_price' 필드가 비어있는 행만 추출
empty_price_rows = df[df['item_price'].isnull()]

# 추출된 행을 따로 저장
empty_price_rows.to_csv('preprocessing_data/3_no_pay_trade_test.csv', index=False)