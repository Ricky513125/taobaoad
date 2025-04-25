import faiss
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Concatenate, Dense
# from tensorflow.keras.models import Model
# 假设已有所有候选物料数据（示例）
item_data = pd.read_csv("data/ad_feature.csv")  # 包含所有需要召回的物料特征

item_tower = tf.keras.models.load_model('models/item_tower.h5')
# 生成物料向量

item_vectors = []
for _, row in item_data.iterrows():
    item_features = {
        'item_adgroup_id': row['adgroup_id'],
        'item_cate_id': row['cate_id'],
        'item_campaign_id': row['campaign_id'],
        'item_customer': row['customer'],
        'item_brand': row['brand'],
        'item_price': row['price'],
        # ...其他物料特征
    }
    vec = item_tower.predict({k: np.array([v]) for k, v in item_features.items()})
    item_vectors.append(vec)

item_vectors = np.concatenate(item_vectors).astype('float32')

# 归一化并构建FAISS索引
faiss.normalize_L2(item_vectors)
index = faiss.IndexFlatIP(item_vectors.shape[1])  # 内积相似度
index.add(item_vectors)

# 保存索引和物料ID映射
faiss.write_index(index, "recall/item_index.faiss")
np.save("recall/item_ids.npy", item_data['item_id'].values)  # 假设物料有唯一ID