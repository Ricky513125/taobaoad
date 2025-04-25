import faiss
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecallService:
    def __init__(self, index_dir: str, user_tower, normalize: bool = True):
        """
        Args:
            index_dir: 索引文件目录
            user_tower: 用户塔模型
            normalize: 是否对向量做归一化
        """
        self.user_tower = user_tower
        self.normalize = normalize
        self.index, self.item_ids = self._load_index(index_dir)
        self.feature_config = self._load_feature_config(index_dir)

    def recall(self, user_features: Dict[str, any], top_k: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """执行召回

        Args:
            user_features: 用户特征字典
            top_k: 召回数量

        Returns:
            (物料ID数组, 相似度分数数组)
        """
        try:
            user_vec = self._get_user_vector(user_features)
            distances, indices = self.index.search(user_vec, top_k)
            return self.item_ids[indices[0]], distances[0]
        except Exception as e:
            logger.error(f"Recall failed: {str(e)}")
            raise

    def _load_index(self, index_dir: str) -> Tuple[faiss.Index, np.ndarray]:
        """加载FAISS索引和物料ID映射"""
        try:
            index_path = f"{index_dir}/item_index.faiss"
            ids_path = f"{index_dir}/item_ids.npy"

            index = faiss.read_index(index_path)
            item_ids = np.load(ids_path)

            logger.info(f"Loaded index with {index.ntotal} items")
            return index, item_ids
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise

    def _get_user_vector(self, user_features: Dict[str, any]) -> np.ndarray:
        """生成用户向量"""
        prepared_features = self._prepare_user_features(user_features)
        user_vec = self.user_tower.predict(prepared_features, verbose=0)

        if self.normalize:
            user_vec = user_vec.astype('float32')
            faiss.normalize_L2(user_vec)

        return user_vec

    def _prepare_user_features(self, raw_features: Dict[str, any]) -> Dict[str, np.ndarray]:
        """准备模型输入特征"""
        features = {}
        for feat_name, feat_value in raw_features.items():
            # 应用特征配置中的类型转换
            feat_type = self.feature_config.get(feat_name, {}).get('type', 'int32')

            if feat_type == 'float32':
                features[feat_name] = np.array([float(feat_value)], dtype='float32')
            else:
                features[feat_name] = np.array([int(feat_value)], dtype='int32')

        return features

    def _load_feature_config(self, index_dir: str) -> Dict:
        """加载特征配置"""
        config_path = Path(index_dir) / "feature_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}