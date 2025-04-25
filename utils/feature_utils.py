from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def build_rank_features(recall_results, user_features, item_features):
    """将召回结果转换为精排特征"""
    rank_data = []

    for user_id, item_ids in recall_results.groupby('user_id')['item_id']:
        user_vec = user_features[user_id]
        user_raw = user_features[user_id]['raw']

        for item_id in item_ids:
            item_vec = item_features[item_id]['vector']
            item_raw = item_features[item_id]['raw']

            # 1. 基础特征
            features = {
            **user_raw,
            **item_raw,
            'cosine_sim': cosine_similarity(user_vec, item_vec),
            'dot_product': np.dot(user_vec, item_vec)
            }

            # 2. 交叉特征（示例）
            features['price_sensitivity'] = user_raw['user_avg_clk_price'] / item_raw['item_price']
            features['cate_match'] = int(item_raw['cate_id'] in user_raw['user_top_cates'])

            rank_data.append(features)

    return pd.DataFrame(rank_data)