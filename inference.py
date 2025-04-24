import numpy as np
import faiss
from model import build_item_tower


def precompute_item_vectors():
    """预计算广告向量"""
    item_tower = tf.keras.models.load_model('item_tower.h5')
    data = pd.read_parquet('processed_data.parquet')

    # 获取所有广告特征
    item_features = {
        'cate_id': data['cate_id'].unique(),
        'brand': data['brand'].unique(),
        'price_bucket': data['price_bucket'].unique()
    }

    vectors = item_tower.predict(item_features)
    np.save('item_vectors.npy', vectors)
    return vectors


def build_faiss_index():
    """构建Faiss索引"""
    vectors = np.load('item_vectors.npy')
    index = faiss.IndexFlatIP(64)  # 内积相似度
    index.add(vectors)
    return index


if __name__ == "__main__":
    precompute_item_vectors()
    index = build_faiss_index()
    faiss.write_index(index, 'item_index.faiss')