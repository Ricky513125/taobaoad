# merge_sequences.py
import pandas as pd



def merge_sequences(data_path, user_seq_path, item_seq_path, output_path):
    """将序列特征合并到原始数据"""
    # 读取原始数据
    data = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)

    # 读取用户行为序列
    user_seqs = pd.read_csv(user_seq_path, index_col='user')['hist_sequence']

    # 读取物品图序列
    item_seqs = pd.read_csv(item_seq_path, header=None, names=['item', 'graph_seq'])
    item_seqs = item_seqs.groupby('item')['graph_seq'].first()

    # 合并用户序列
    data['hist_cate_seq'] = data['user'].map(user_seqs)
    data['hist_brand_seq'] = data['user'].map(user_seqs)  # 实际应用中可分开处理

    # 合并物品图序列
    data['graph_seq'] = data['item'].map(item_seqs)

    # 填充缺失值
    data.fillna({'hist_cate_seq': '', 'hist_brand_seq': '', 'graph_seq': ''}, inplace=True)

    # 保存处理后的数据
    data.to_parquet(output_path)


if __name__ == "__main__":
    merge_sequences(
        data_path="data/processed_data.parquet",
        user_seq_path="data/user_behavior_sequences.csv",
        item_seq_path="data/item_graph_sequences.csv",
        output_path="data/processed_data_with_sequences.parquet"
    )