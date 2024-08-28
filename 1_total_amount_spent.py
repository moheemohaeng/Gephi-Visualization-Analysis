import pandas as pd

##### 유저별 월간 결제 금액을 통합 ##### clear

df = pd.read_csv('source_data/test1_payment.csv')

# acc_id 별로 amount_spent 합계 계산
total_amount_spent = df.groupby('acc_id')['amount_spent'].sum().reset_index()

# 결과 저장
total_amount_spent.to_csv('preprocessing_data/1_total_amount_spent_test.csv', index=False)