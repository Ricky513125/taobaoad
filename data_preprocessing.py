import pandas as pd
import numpy as np


def load_and_preprocess():
    """加载并预处理数据"""
    # 加载原始数据
    ad_feature = pd.read_csv('data/ad_feature.csv')
    user_profile = pd.read_csv('data/user_profile.csv')
    raw_sample = pd.read_csv('data/raw_sample_train.csv')

    # 合并数据
    data = pd.merge(raw_sample, user_profile, left_on='user', right_on='userid')
    data = pd.merge(data, ad_feature, on='adgroup_id')

    # 处理缺失值
    data['pvalue_level'] = data['pvalue_level'].fillna(0).astype(int)  # 确保整数类型 # -1表示未知
    print(data.columns.tolist())
    # data['new_user_class_level'].fillna(0, inplace=True)
    data['new_user_class_level'] = data['new_user_class_level'].fillna(0).astype(int)

    # 处理异常值
    data['price'] = np.where(data['price'] <= 0, data['price'].median(), data['price'])

    # 特征分桶
    def bucketize(series, num_buckets=10):
        return pd.qcut(series, num_buckets, labels=False, duplicates='drop')

    data['price_bucket'] = bucketize(data['price'], num_buckets=10)
    return data


if __name__ == "__main__":
    data = load_and_preprocess()
    data.to_parquet('processed_data.parquet')  # 保存预处理数据