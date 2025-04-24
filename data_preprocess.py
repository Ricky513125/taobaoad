import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载四个数据集
# data_sets = {
#     "raw_sample": pd.read_csv("data/raw_sample.csv"),
#     "ad_feature": pd.read_csv("data/ad_feature.csv"),
#     "user_profile": pd.read_csv("data/user_profile.csv"),
#     "behavior_log": pd.read_csv("data/behavior_log.csv")  # 抽样1%行为日志, sample_frac=0.01
# }

raw_sample = pd.read_csv("data/raw_sample.csv")
# ad_feature = pd.read_csv("data/ad_feature.csv")
# user_profile = pd.read_csv("data/user_profile.csv")
behavior_log = pd.read_csv("data/behavior_log.csv")

cleaned_behavior = behavior_log[behavior_log['time_stamp'] > 0]
cleaned_behavior['tag_weight'] = np.where(cleaned_behavior['btag'] == 'pv', 1, np.where(cleaned_behavior['btag'] == 'cart', 2, np.where(cleaned_behavior['btag'] == 'fav', 3, 4)))
cleaned_behavior.to_csv('data/cleaned_behavior.csv', index=False)


## raw_sample
# 定义时间边界（使用原始int64时间戳）
train_start = 1494028800  # 2017-05-06 00:00:00
train_end = 1494547200    # 2017-05-12 23:59:59
test_start = 1494547200   # 2017-05-13 00:00:00

# 划分数据集（不转换时间戳格式）
train_data = behavior_log[
    (raw_sample['time_stamp'] >= train_start) &
    (raw_sample['time_stamp'] <= train_end)
]
test_data = raw_sample[raw_sample['time_stamp'] >= test_start]

# 验证时间戳格式
print("训练集时间戳类型:", train_data['time_stamp'].dtype)  # 应输出 int64
print("测试集时间戳类型:", test_data['time_stamp'].dtype)  # 应输出 int64

# 保存数据集（保持原始格式）
train_data.to_csv('data/raw_sample_train.csv', index=False)
test_data.to_csv('data/raw_sample_test.csv', index=False)
