import os
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss
import json
from tqdm import tqdm
"""
针对4.27修改过的数据集进行重新的普通双塔召回评估

4.28 
100 个召回全为0 
增加500 和1000召回量看看

4.29 
添加单用户验证， 检测能否召回到
"""

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
        """执行评估流程（添加进度条）"""
        os.makedirs(output_dir, exist_ok=True)

        # 读取测试数据并显示进度
        print("正在加载测试数据...")
        test_data = pd.read_parquet(test_data_path)
        user_groups = test_data.groupby('user_id')
        total_users = len(user_groups)
        print(f"共需处理 {total_users} 个用户")

        # 在_evaluate方法中添加调试代码
        print("\n=== 调试信息 ===")
        print(f"测试数据用户数: {len(user_groups)}")
        print(f"首用户ID: {next(iter(user_groups.groups))}")
        print(f"首用户真实物品: {user_groups.get_group(next(iter(user_groups.groups)))['adgroup_id'].values[:5]}")
        print(f"FAISS索引维度: {self.index.d}")
        # print(f"用户向量示例: {user_vectors[0][:5] if len(user_vectors) > 0 else '无'}")

        # 检查用户画像与测试数据对齐
        test_users = set(test_data['user_id'].unique())
        profile_users = set(pd.read_parquet("data/user.parquet")['userid'])
        print(f"测试集用户与画像重叠率: {len(test_users & profile_users) / len(test_users):.1%}")


        # 初始化结果存储
        results = {f'top_{k}': {
            'hit_rate': [],
            'precision': [],
            'ndcg': [],
            'user_samples': []
        } for k in top_k_list}

        # 分批处理用户（添加主进度条）
        batch_size = 1024
        with tqdm(total=total_users, desc="总体进度") as pbar:
            for batch_users in self._batch_users(user_groups, batch_size):
                # GPU批量生成用户向量
                with tf.device('/GPU:0'):
                    user_vectors = self._generate_user_vectors_batch(batch_users)

                # CPU执行召回（添加批次内进度条）
                batch_user_ids = list(batch_users.keys())
                for i, (user_id, vector) in enumerate(zip(batch_user_ids, user_vectors)):
                    user_data = batch_users[user_id]
                    true_items = user_data['adgroup_id'].values

                    # 处理每个top_k（移除嵌套进度条改为简单循环）
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

                        # 更新主进度条
                        pbar.update(1)
                        pbar.set_postfix({
                            '当前批次': f"{i + 1}/{len(batch_user_ids)}",
                            '最新命中率': f"{metrics['hit']:.0%}"
                        })

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

    def generate_rerank_input(self, test_data_path, top_k=100):
        """ 生成DeepFM输入数据 """
        test_data = pd.read_parquet(test_data_path)
        user_groups = test_data.groupby('user')

        rerank_data = []
        for user_id, group in tqdm(user_groups, desc="生成精排输入"):
            # 获取召回结果
            user_feat = self.user_profile.get_user_features(user_id)
            with tf.device('/GPU:0'):
                user_vec = self.user_tower.predict(np.array([list(user_feat.values())]))

            _, indices = self.index.search(user_vec.astype('float32'), top_k)
            candidate_items = self.item_ids[indices[0]]

            # 存储精排输入
            rerank_data.append({
                'user_id': user_id,
                'candidate_items': candidate_items.tolist(),
                'true_items': group['adgroup_id'].tolist()
            })

        return pd.DataFrame(rerank_data)

    def check_vector_spaces(self):
        """检查双塔输出空间是否匹配"""
        # 随机选取5个用户和物品
        user_sample = self.user_tower.predict(np.random.rand(5, 8))  # 假设8个特征
        item_sample = self.item_tower.predict(np.random.rand(5, 6))  # 假设6个特征

        print("\n=== 向量空间验证 ===")
        print(f"用户向量均值: {np.mean(user_sample):.4f} ± {np.std(user_sample):.4f}")
        print(f"物品向量均值: {np.mean(item_sample):.4f} ± {np.std(item_sample):.4f}")
        print(
            f"余弦相似度范围: {np.dot(user_sample, item_sample.T).min():.2f} - {np.dot(user_sample, item_sample.T).max():.2f}")

class UserProfileAccessor:
    """用户画像访问器（与之前相同）"""

    def __init__(self, profile_path):
        self.profile_path = profile_path
        self._init_data_source()

    def _init_data_source(self):
        if self.profile_path.endswith('.parquet'):
            self.profile_df = pd.read_parquet(self.profile_path)
            self.profile_df.set_index('userid', inplace=True)
            # self.get_fn = self._get_from_dataframe
        else:
            raise ValueError("仅支持parquet文件")

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
                'occupation': int(user_data['occupation'])
            }
        except KeyError:
            print(f"警告：用户 {user_id} 不在画像数据中")
            return None

if __name__ == "__main__":
    # 初始化评估器
    evaluator = RecallEvaluator(
        user_tower_path="results/0427embed/user_tower",
        item_tower_path="results/0427embed/item_tower",
        item_index_path="recall/item_index_1745730778.faiss",
        item_ids_path="recall/item_ids_1745730778.npy",
        user_profile_path="data/user.parquet"
    )

    evaluator.check_vector_spaces()
    # # 第三步：验证单个用户召回
    # test_user_id = 1  # 使用您调试信息中的首用户ID
    # test_items = [133190, 142774, 769066]  # 该用户的真实物品
    #
    # user_feat = evaluator.user_profile.get_user_features(test_user_id)
    # user_vec = evaluator.user_tower.predict(np.array([list(user_feat.values())]))
    #
    # distances, indices = evaluator.index.search(user_vec.astype('float32'), 1000)
    # recalled = evaluator.item_ids[indices[0]]
    # print("\n=== 单用户验证 ===")
    # print(f"用户ID: {test_user_id}")
    # print(f"真实物品: {test_items}")
    # print(f"召回物品交集: {set(recalled) & set(test_items)}")
    # print(f"Top100召回示例: {recalled[:100]}")


    # 执行评估（添加外层进度描述）
    print("=" * 50)
    print("开始双塔召回评估")
    print("=" * 50)
    # 执行评估
    results = evaluator.evaluate(
        test_data_path="data/processed_data_test3.parquet",
        top_k_list=[10, 20, 50, 100, 500, 1000],
        output_dir="recall/results/recall_4_29"
    )

    # 打印结果摘要
    print("\n召回测试结果摘要:")
    for top_k, metrics in results.items():
        print(f"\n--- {top_k} ---")
        print(f"用户数: {metrics['num_users']}")
        print(f"平均命中率: {metrics['hit_rate']:.2%}")
        print(f"平均精度: {metrics['precision']:.4f}")
        print(f"平均NDCG: {metrics['ndcg']:.4f}")