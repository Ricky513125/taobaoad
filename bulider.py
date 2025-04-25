# 向量索引构建
import faiss
import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorIndexBuilder:
    def __init__(self, item_tower, batch_size: int = 1024):
        """
        Args:
            item_tower: 预加载的物料塔模型
            batch_size: 批量处理大小
        """
        self.item_tower = item_tower
        self.batch_size = batch_size

    def build_index(self, item_data: pd.DataFrame, output_dir: str, normalize: bool = True) -> None:
        """构建并保存FAISS索引

        Args:
            item_data: 包含物料特征的DataFrame
            output_dir: 索引输出目录
            normalize: 是否对向量做L2归一化
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 生成物料向量
        vectors = self._generate_vectors(item_data)

        # 向量归一化
        if normalize:
            faiss.normalize_L2(vectors)

        # 创建并保存索引
        self._save_index(vectors, item_data['item_id'].values, output_dir)
        logger.info(f"Index built with {len(vectors)} items")

    def _generate_vectors(self, item_data: pd.DataFrame) -> np.ndarray:
        """批量生成物料向量"""
        item_features = self._prepare_features(item_data)
        vectors = []

        for i in tqdm(range(0, len(item_data), self.batch_size), desc="Generating vectors"):
            batch = {k: v[i:i + self.batch_size] for k, v in item_features.items()}
            batch_vec = self.item_tower.predict(batch, verbose=0)
            vectors.append(batch_vec)

        return np.concatenate(vectors).astype('float32')

    def _prepare_features(self, item_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """准备模型输入特征"""
        return {
            'item_adgroup_id': item_data['adgroup_id'].values,
            'item_cate_id': item_data['cate_id'].values,
            'item_campaign_id': item_data['campaign_id'].values.astype('int32'),
            'item_customer': item_data['customer'].values,
            'item_brand': item_data['brand'].values.astype('float32'),
            'item_price': item_data['price'].values.astype('float32')
        }

    def _save_index(self, vectors: np.ndarray, item_ids: np.ndarray, output_dir: str) -> None:
        """保存索引和物料ID映射"""
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        faiss.write_index(index, f"{output_dir}/item_index.faiss")
        np.save(f"{output_dir}/item_ids.npy", item_ids)
        logger.info(f"Index saved to {output_dir}")