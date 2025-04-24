import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 配置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# def load_data(file_path, sample_frac=0.1):
#     """加载数据并抽样（适用于大文件）"""
#     if "behavior_log" in file_path:
#         return pd.read_csv(file_path,
#                            nrows=int(7e8 * sample_frac) if sample_frac < 1 else None,
#                            parse_dates=['time_stamp'])
#     return pd.read_csv(file_path)


# 加载四个数据集
data_sets = {
    # "raw_sample": pd.read_csv("data/raw_sample.csv"),
    # "ad_feature": pd.read_csv("data/ad_feature.csv"),
    # "user_profile": pd.read_csv("data/user_profile.csv"),
    "behavior_log": pd.read_csv("data/behavior_log.csv")  # 抽样1%行为日志, sample_frac=0.01
}


# ======================
# 一、数据概览分析
# ======================
def data_overview(df, name):
    """生成数据概览报告"""
    print(f"\n===== {name} 数据概览 =====")

    # 基础信息
    print("\n【基础信息】")
    print(f"记录数：{len(df):,}")
    print(f"字段数：{len(df.columns)}")
    print("\n字段类型：")
    print(df.dtypes.to_frame().rename(columns={0: 'dtype'}))

    # 头部数据展示
    print("\n【头部数据】")
    print(df.head(2))

    # 数值型字段统计
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        print("\n【数值型字段统计】")
        print(df[num_cols].describe())
    # print("max", max())

    # 分类字段统计
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].nunique() < 20:
            print(f"\n【{col} 值分布】")
            print(df[col].value_counts(dropna=False).to_frame())


# 执行数据概览
for name, df in data_sets.items():
    data_overview(df.copy(), name)


# ======================
# 二、空值检测
# ======================
def missing_analysis(df, name):
    """空值分析"""
    print(f"\n===== {name} 空值分析 =====")
    missing = df.isnull().sum().to_frame(name='missing_count')
    missing['missing_ratio'] = missing['missing_count'] / len(df)
    print(missing[missing['missing_count'] > 0])


for name, df in data_sets.items():
    missing_analysis(df, name)


# ======================
# 三、异常值检测
# ======================
def detect_anomalies(df, name):
    """异常值检测"""
    print(f"\n===== {name} 异常值检测 =====")

    if name == "raw_sample":
        # 检查点击标记一致性
        inconsistent_clk = df[(df['clk'] + df['noclk']) != 1]
        print(f"\n点击标记矛盾记录数：{len(inconsistent_clk)}")

        # 时间戳范围检查
        time_min = df['time_stamp'].min()
        time_max = df['time_stamp'].max()
        print(f"时间戳范围：{time_min} ~ {time_max}")

    elif name == "ad_feature":
        # 价格异常检测
        price_stats = df['price'].describe()
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        df['price'].plot(kind='box')
        plt.subplot(122)
        df[df['price'] < 1000]['price'].hist(bins=50)
        plt.suptitle('Price Distribution')
        plt.show()

        # 负值价格
        neg_prices = df[df['price'] < 0]
        print(f"负价格记录数：{len(neg_prices)}")

    elif name == "user_profile":
        # 分类变量范围检查
        validations = {
            'final_gender_code': [1, 2],
            'age_level': [1, 2, 3, 4, 5, 6, 7, 8],  # 假设8个年龄段
            'pvalue_level': [1, 2, 3],
            'shopping_level': [1, 2, 3],
            'occupation': [0, 1]
        }

        for col, valid_values in validations.items():
            invalid = ~df[col].isin(valid_values)
            print(f"{col} 异常值数量：{invalid.sum()}")

    elif name == "behavior_log":
        # 行为类型验证
        valid_btags = ['pv', 'cart', 'fav', 'buy']
        invalid_btag = ~df['btag'].isin(valid_btags)
        print(f"异常行为类型数量：{invalid_btag.sum()}")


# for name, df in data_sets.items():
#     detect_anomalies(df.copy(), name)

# ======================
# 四、关联性检查
# ======================
print("\n===== 跨表关联性检查 =====")

# 检查广告ID一致性
# ad_intersect = len(set(data_sets['raw_sample']['adgroup_id']).intersection(
#     set(data_sets['ad_feature']['adgroup_id'])))
# print(f"广告ID匹配率：{ad_intersect / len(data_sets['raw_sample']):.2%}")
#
# # 检查用户ID一致性
# user_intersect = len(set(data_sets['raw_sample']['user_id']).intersection(
#     set(data_sets['user_profile']['userid'])))
# print(f"用户ID匹配率：{user_intersect / len(data_sets['raw_sample']):.2%}")
#
# # 行为日志时间跨度检查
# behavior_days = (data_sets['behavior_log']['time_stamp'].max() -
#                  data_sets['behavior_log']['time_stamp'].min()).days
# print(f"行为日志时间跨度：{behavior_days}天（应为22天）")