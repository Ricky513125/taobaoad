"""
将用户信息、广告信息、 用户点击信息合并到一张表，便于后面进行处理


"""
import pandas as pd
import numpy as np

user = pd.read_parquet('data/user.parquet')
ad = pd.read_csv('data/ad_feature.csv')
train = pd.read_parquet('data/processed_data_train.parquet')
test = pd.read_parquet('data/processed_data_test.parquet')

combined_train = pd.merge(train, ad, how='left', on='adgroup_id')
combined_train = pd.merge(combined_train, user, how='left', left_on='user', right_on='userid')
combined_train.to_parquet('data/combined_train.parquet')

combined_test = pd.merge(test, ad, how='left', on='adgroup_id')
combined_test = pd.merge(combined_test, user, how='left', left_on='user', right_on='userid')
combined_test.to_parquet('data/combined_test.parquet')
print("Successfully!")
