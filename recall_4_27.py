import os
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss
import json
from tqdm import tqdm


class RecallEvaluator:
    def __init__(self, user_tower_path, item_tower_path, item_index_path, item_ids_path, user_profile_path):
        """
        混合加速评估器初始化
        - TensorFlow模型使用GPU
        - FAISS使用CPU
        """
        # 配置TensorFlow GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"已启用GPU加速：{gpus}")
            except RuntimeError as e:
                print(e)

        # 加载模型（强制使用GPU）
        with tf.device('/GPU:0'):
            self.user_tower = tf.keras.models.load_model(user_tower_path)
            self.item_tower = tf.keras.models.load_model(item_tower_path)

        # 加载FAISS CPU索引
        self.index = faiss.read_index(item_index_path)
        self.item_ids = np.load(item_ids_path)
        print(f"FAISS索引已加载（CPU模式），包含 {self.index.ntotal} 个向量")

        # 初始化用户画像访问器
        self.user_profile = UserProfileAccessor(user_profile_path)

    def evaluate(self, test_data_path, top_k_list=[10, 20, 50], output_dir="recall/results"):
        """执行评估流程"""
        os.makedirs(output_dir, exist_ok=True)
        test_data = pd.read_parquet(test_data_path)
        user_groups = test_data.groupby('user_id')

        # 预分配GPU内存
        batch_size = 1024
        sample_user = next(iter(user_groups))[1].iloc[0]
        user_vector_dim = self._generate_user_vector(
            self.user_profile.get_user_features(sample_user['user_id'])
        ).shape[0]

        # 初始化结果存储
        results = {f'top_{k}': {
            'hit_rate': [],
            'precision': [],
            'ndcg': [],
            'user_samples': []
        } for k in top_k_list}

        # 分批处理用户
        for batch_users in self._batch_users(user_groups, batch_size):
            # GPU批量生成用户向量
            with tf.device('/GPU:0'):
                user_vectors = self._generate_user_vectors_batch(batch_users)

            # CPU执行召回
            for user_id, vector in zip(batch_users.keys(), user_vectors):
                user_data = batch_users[user_id]
                true_items = user_data['adgroup_id'].values

                for k in top_k_list:
                    # FAISS CPU搜索
                    distances, indices = self.index.search(
                        np.expand_dims(vector, axis=0).astype('float32'),
                        k
                    )
                    pred_items = self.item_ids[indices[0]]

                    # 计算指标
                    metrics = self._calculate_metrics(true_items, pred_items, k)

                    # 存储结果
                    if len(results[f'top_{k}']['user_samples']) < 10:
                        results[f'top_{k}']['user_samples'].append({
                            'user_id': user_id,
                            'true_items': true_items.tolist(),
                            'pred_items': pred_items.tolist(),
                        **metrics
                        })
                        results[f'top_{k}']['hit_rate'].append(metrics['hit'])
                        results[f'top_{k}']['precision'].append(metrics['precision'])
                        results[f'top_{k}']['ndcg'].append(metrics['ndcg'])

            # 汇总结果
            final_metrics = self._aggregate_results(results, top_k_list)
            self._save_results(final_metrics, output_dir)
            return final_metrics

        def _init_gpu(self):
            """初始化GPU资源"""
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"GPU已启用：{gpus}")
                    return True
                except RuntimeError as e:
                    print(f"GPU配置错误：{e}")
            return False

        def _batch_users(self, user_groups, batch_size):
            """用户分批生成器"""
            batch = {}
            for user_id, group in user_groups:
                batch[user_id] = group
                if len(batch) >= batch_size:
                    yield batch
                    batch = {}
            if batch:
                yield batch

        def _generate_user_vectors_batch(self, batch_users):
            """GPU批量生成用户向量"""
            # 准备批量输入
            batch_features = []
            for user_id, group in batch_users.items():
                features = self.user_profile.get_user_features(user_id)
                if features:
                    batch_features.append(features)

            # 转换为模型输入格式
            inputs = {
                'user_cms_segid': np.array([f['cms_segid'] for f in batch_features], dtype=np.int32),
                'user_cms_group_id': np.array([f['cms_group_id'] for f in batch_features], dtype=np.int32),
                'user_gender': np.array([f['final_gender_code'] for f in batch_features], dtype=np.int32),
                'user_age_level': np.array([f['age_level'] for f in batch_features], dtype=np.int32),
                'user_pvalue_level': np.array([f['pvalue_level'] + 1 for f in batch_features], dtype=np.int32),
                'user_shopping_level': np.array([f['shopping_level'] for f in batch_features], dtype=np.int32),
                'user_city_level': np.array([f['new_user_class_level'] for f in batch_features], dtype=np.int32),
                'user_occupation': np.array([f['occupation'] for f in batch_features], dtype=np.float32)
            }

            # GPU预测
            return self.user_tower.predict(inputs, batch_size=len(batch_features))

        def _calculate_metrics(self, true_items, pred_items, k):
            """计算评估指标"""
            hits = np.where(np.isin(pred_items, true_items))[0]
            if len(hits) == 0:
                return {'hit': 0, 'precision': 0, 'ndcg': 0}

            first_hit_rank = hits[0]
            return {
                'hit': 1,
                'precision': 1.0 / (first_hit_rank + 1),
                'ndcg': 1.0 / np.log2(first_hit_rank + 2)
            }

        def _aggregate_results(self, results, top_k_list):
            """聚合结果"""
            return {
                f'top_{k}': {
                    'hit_rate': np.mean(results[f'top_{k}']['hit_rate']),
                    'precision': np.mean(results[f'top_{k}']['precision']),
                    'ndcg': np.mean(results[f'top_{k}']['ndcg']),
                    'num_users': len(results[f'top_{k}']['hit_rate']),
                    'sample_results': results[f'top_{k}']['user_samples']
                } for k in top_k_list
            }

        def _save_results(self, metrics, output_dir):
            """保存结果"""
            with open(f"{output_dir}/metrics.json", 'w') as f:
                json.dump({
                    k: {m: v for m, v in v.items() if m != 'sample_results'}
                    for k, v in metrics.items()
                }, f, indent=2)

            with open(f"{output_dir}/sample_details.json", 'w') as f:
                json.dump({
                    k: v['sample_results']
                    for k, v in metrics.items()
                }, f, indent=2)
            print(f"结果已保存到 {output_dir}")

class UserProfileAccessor:
    """用户画像访问器（与之前相同）"""

    def __init__(self, profile_path):
        self.profile_path = profile_path
        self._init_data_source()

    def _init_data_source(self):
        if self.profile_path.endswith('.csv'):
            self.profile_df = pd.read_csv(self.profile_path)
            self.profile_df.set_index('user_id', inplace=True)
            self.get_fn = self._get_from_dataframe
        else:
            raise ValueError("仅支持CSV文件")

    def get_user_features(self, user_id):
        try:
            user_data = self.profile_df.loc[user_id]
            return {
                'cms_segid': int(user_data['cms_segid']),
                'cms_group_id': int(user_data['cms_group_id']),
                'final_gender_code': int(user_data['final_gender_code']),
                'age_level': int(user_data['age_level']),
                'pvalue_level': int(user_data['pvalue_level']),
                'shopping_level': int(user_data['shopping_level']),
                'new_user_class_level': int(user_data['new_user_class_level']),
                'occupation': float(user_data['occupation'])
            }
        except KeyError:
            print(f"警告：用户 {user_id} 不在画像数据中")
            return None

if __name__ == "__main__":
    # 初始化评估器
    evaluator = RecallEvaluator(
        user_tower_path="user_tower",
        item_tower_path="item_tower",
        item_index_path="recall/item_index_1745730778.faiss",
        item_ids_path="recall/item_ids_1745730778.npy",
        user_profile_path="data/user_profile.csv"
    )

    # 执行评估
    results = evaluator.evaluate(
        test_data_path="processed_data_test.parquet",
        top_k_list=[10, 20, 50],
        output_dir="recall/results"
    )

    # 打印结果摘要
    print("\n召回测试结果摘要:")
    for top_k, metrics in results.items():
        print(f"\n--- {top_k} ---")
        print(f"用户数: {metrics['num_users']}")
        print(f"平均命中率: {metrics['hit_rate']:.2%}")
        print(f"平均精度: {metrics['precision']:.4f}")
        print(f"平均NDCG: {metrics['ndcg']:.4f}")