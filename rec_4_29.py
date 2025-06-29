import os
import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_auc_score, precision_score
import tensorflow_ranking as tfr

"""
完整版的处理流程
带ID
"""

# 配置参数
class Config:
    # 路径配置
    data_dir = Path("data")
    seq_dir = Path("data/sequence")
    model_dir = Path("models")
    result_dir = Path("results/0429rec")

    # 召回参数
    recall_top_k = 100
    user_seq_len = 20  # 用户行为序列长度

    # 模型参数
    user_embed_dim = 64
    item_embed_dim = 64
    lstm_units = 32
    item_vocab_size = 10000  # 物品ID总数

    # 训练参数
    batch_size = 32
    epochs = 10


# 数据处理器
class DataProcessor:
    def __init__(self, config):
        self.cfg = config
        self.user_sequences = None
        self.item_graph = None
        self.train_data = None
        self.test_data = None

        # 特征列定义
        self.user_feature_cols = [
            'user_id', 'cms_segid', 'cms_group_id', 'final_gender_code',
            'age_level', 'pvalue_level', 'shopping_level',
            'occupation', 'new_user_class_level'
        ]
        self.item_feature_cols = [
            'adgroup_id', 'cate_id', 'campaign_id', 'brand', 'price'
        ]

    def load_all_data(self):
        """加载所有必要数据"""
        # 加载主数据
        self.train_data = pd.read_parquet(self.cfg.data_dir / 'processed_data_train.parquet')
        self.test_data = pd.read_parquet(self.cfg.data_dir / 'processed_data_test3.parquet')

        # 加载序列数据
        seq_path = self.cfg.seq_dir / "user_sequences_optimized.csv"

        # 方法1：使用pandas读取，处理不定长序列
        try:
            # 读取CSV，确保识别header
            seq_df = pd.read_csv(seq_path)
            print("成功读取CSV，检测到列名:", seq_df.columns.tolist())

            # 验证列名
            if 'user_id' not in seq_df.columns or 'hist_sequence' not in seq_df.columns:
                raise ValueError("CSV必须包含user_id和hist_sequence列")

            # 解析序列函数（处理不定长）
            def parse_sequence(row):
                try:
                    user_id = int(row['user_id'])
                    seq_pairs = []
                    if pd.notna(row['hist_sequence']):
                        for pair in row['hist_sequence'].split('|'):
                            if pair and ',' in pair:
                                cate, brand = map(int, pair.split(',', 1))
                                seq_pairs.append((cate, brand))
                    return user_id, seq_pairs
                except Exception as e:
                    print(f"解析错误 行内容:{row} 错误:{str(e)}")
                    return None, []

            # 应用解析
            parsed_data = []
            for _, row in tqdm(seq_df.iterrows(), total=len(seq_df), desc="解析序列"):
                user_id, seq = parse_sequence(row)
                if user_id is not None:
                    parsed_data.append((user_id, seq))

            # 转换为字典
            self.user_sequences = {
                user_id: seq[:self.cfg.user_seq_len]
                for user_id, seq in parsed_data
            }

        except Exception as e:
            print(f"Pandas读取失败: {str(e)}")
            # 方法2：手动逐行读取（兼容性更强）
            self.user_sequences = {}
            with open(seq_path, 'r') as f:
                # 跳过header
                next(f)
                for line in tqdm(f, desc="手动解析序列"):
                    try:
                        # 分割第一个逗号前的user_id和后面的序列
                        parts = line.strip().split(',', 1)
                        if len(parts) < 2:
                            continue

                        user_id = int(parts[0])
                        seq_pairs = []

                        # 处理可能为空的序列
                        if parts[1].strip():
                            for pair in parts[1].split('|'):
                                if pair and ',' in pair:
                                    cate, brand = map(int, pair.split(',', 1))
                                    seq_pairs.append((cate, brand))

                        self.user_sequences[user_id] = seq_pairs[:self.cfg.user_seq_len]
                    except Exception as e:
                        print(f"行解析失败: {line[:50]}... 错误: {str(e)}")

        # # 加载图嵌入
        # graph_path = self.cfg.data_dir / "item_graph_optimized.pkl"
        # with open(graph_path, 'rb') as f:
        #     self.item_graph = pickle.load(f)

            # 修改图加载部分 - 确保加载的是共现字典而不是DiGraph
            graph_path = self.cfg.data_dir / "item_graph_optimized.pkl"
            with open(graph_path, 'rb') as f:
                graph_data = pickle.load(f)

                # 检查加载的数据类型
                if isinstance(graph_data, defaultdict):
                    self.item_graph = graph_data
                else:
                    # 如果是DiGraph，转换为字典格式
                    self.item_graph = self._convert_graph_to_dict(graph_data)

        # 数据预处理
        self._preprocess_data()

    def _convert_graph_to_dict(self, graph):
        """将NetworkX图转换为共现字典格式"""
        from collections import defaultdict, Counter
        co_occur = defaultdict(Counter)

        for u, v, data in graph.edges(data=True):
            co_occur[u][v] = data.get('weight', 1)

        return co_occur

    def _preprocess_data(self):
        """数据预处理（调整为处理cate,brand对）"""
        # 数值特征标准化
        numeric_cols = ['price', 'pvalue_level', 'shopping_level']
        for col in numeric_cols:
            if col in self.train_data.columns:
                mean = self.train_data[col].mean()
                std = self.train_data[col].std()
                self.train_data[col] = (self.train_data[col] - mean) / std
                self.test_data[col] = (self.test_data[col] - mean) / std

        # # 填充缺失的图嵌入
        # default_graph_emb = np.zeros(64)  # 假设图嵌入维度为64
        # self.train_data['graph_emb'] = self.train_data['adgroup_id'].apply(
        #     lambda x: self.item_graph.get(x, default_graph_emb)
        # )
        # self.test_data['graph_emb'] = self.test_data['adgroup_id'].apply(
        #     lambda x: self.item_graph.get(x, default_graph_emb)
        # )

            # 修改图嵌入处理部分
            default_graph_emb = np.zeros(64)  # 假设图嵌入维度为64

            def get_graph_emb(x):
                # 生成与共现图中一致的物品标识
                item_id = f"{x['cate_id']}_{x['brand']}" if isinstance(x, pd.Series) else f"{x}_0"

                # 从共现字典中获取相关项
                related_items = self.item_graph.get(item_id, {})

                if not related_items:
                    return default_graph_emb

                # 简单示例：取top3相关项的均值作为嵌入
                top_items = sorted(related_items.items(), key=lambda x: -x[1])[:3]
                emb = np.mean([self._get_item_embedding(item) for item, _ in top_items], axis=0)
                return emb if not np.isnan(emb).any() else default_graph_emb

            # 临时解决方案：如果没有单独的嵌入，使用one-hot编码
            unique_items = set().union(*[v.keys() for v in self.item_graph.values()]).union(self.item_graph.keys())
            item_to_idx = {item: i for i, item in enumerate(unique_items)}
            self._get_item_embedding = lambda x: np.eye(len(item_to_idx))[item_to_idx[x]]

            self.train_data['graph_emb'] = self.train_data.apply(
                lambda row: get_graph_emb(row), axis=1
            )
            self.test_data['graph_emb'] = self.test_data.apply(
                lambda row: get_graph_emb(row), axis=1
            )

        # 添加序列数据（调整为处理cate,brand对）
        def process_sequence(seq_pairs):
            """处理cate,brand序列对"""
            if not seq_pairs:
                return [0] * self.cfg.user_seq_len * 2  # 每个对包含两个元素

            # 展平序列对 (c1,b1,c2,b2,...)
            flat_seq = []
            for pair in seq_pairs[:self.cfg.user_seq_len]:
                flat_seq.extend([int(pair[0]), int(pair[1])])

            # 填充不足部分
            if len(flat_seq) < self.cfg.user_seq_len * 2:
                flat_seq += [0] * (self.cfg.user_seq_len * 2 - len(flat_seq))

            return flat_seq

        self.train_data['hist_seq'] = self.train_data['user_id'].apply(
            lambda x: process_sequence(self.user_sequences.get(x, []))
        )
        self.test_data['hist_seq'] = self.test_data['user_id'].apply(
            lambda x: process_sequence(self.user_sequences.get(x, []))
        )

    def _pad_sequence(self, seq):
        """填充序列到固定长度"""
        if len(seq) >= self.cfg.user_seq_len:
            return seq[:self.cfg.user_seq_len]
        else:
            return seq + [0] * (self.cfg.user_seq_len - len(seq))


# 召回模型
class SequenceRecallModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.user_encoder = self._build_user_tower()
        self.item_encoder = self._build_item_tower()

    def _build_user_tower(self):
        """用户塔：基础特征+行为序列"""
        # 基础特征输入
        base_input = Input(shape=(len(self.cfg.user_feature_cols) - 1,))  # 排除user_id

        # 序列特征输入
        seq_input = Input(shape=(self.cfg.user_seq_len,))
        seq_emb = Embedding(self.cfg.item_vocab_size, 32)(seq_input)
        seq_feat = LSTM(self.cfg.lstm_units)(seq_emb)

        # 合并特征
        concat = Concatenate()([base_input, seq_feat])
        output = Dense(self.cfg.user_embed_dim)(concat)
        return Model(inputs=[base_input, seq_input], outputs=output)

    def _build_item_tower(self):
        """物品塔：基础特征+图嵌入"""
        # 基础特征输入
        base_input = Input(shape=(len(self.cfg.item_feature_cols) - 1,))  # 排除item_id

        # 图嵌入输入
        graph_input = Input(shape=(64,))  # 假设图嵌入维度64

        # 合并特征
        concat = Concatenate()([base_input, graph_input])
        output = Dense(self.cfg.item_embed_dim)(concat)
        return Model(inputs=[base_input, graph_input], outputs=output)

    def call(self, inputs):
        user_emb = self.user_encoder([inputs['user_base'], inputs['user_seq']])
        item_emb = self.item_encoder([inputs['item_base'], inputs['item_graph']])
        return tf.linalg.matmul(user_emb, item_emb, transpose_b=True)


# 召回处理器
class RecallHandler:
    def __init__(self, config, processor):
        self.cfg = config
        self.processor = processor
        self.result_dir = config.result_dir / "recall"
        os.makedirs(self.result_dir, exist_ok=True)

    def generate_recalls(self, model):
        """生成召回结果"""
        # 准备物品池
        item_pool = self.processor.train_data.drop_duplicates('adgroup_id')

        # 生成物品嵌入
        item_embs = model.item_encoder.predict(
            {
                'item_base': item_pool[[c for c in self.processor.item_feature_cols if c != 'adgroup_id']].values,
                'item_graph': item_pool['graph_emb'].values
            },
            batch_size=self.cfg.batch_size
        )

        # 为每个用户生成召回
        for user_id, group in tqdm(self.processor.train_data.groupby('user_id'), desc="生成召回"):
            user_emb = model.user_encoder.predict(
                {
                    'user_base': group[[c for c in self.processor.user_feature_cols if c != 'user_id']].values[:1],
                    'user_seq': np.array(group['hist_seq'].values[:1])
                },
                verbose=0
            )

            # 计算相似度
            scores = np.dot(user_emb, item_embs.T)[0]
            top_k = np.argsort(scores)[-self.cfg.recall_top_k:][::-1]

            # 保存结果
            with open(self.result_dir / f"{user_id}.json", 'w') as f:
                json.dump({
                    "user_id": int(user_id),
                    "recalls": item_pool.iloc[top_k]['adgroup_id'].tolist(),
                    "scores": scores[top_k].tolist()
                }, f)

    def evaluate_recall(self):
        """评估召回效果"""
        hit_counts = []
        for user_id, group in tqdm(self.processor.test_data.groupby('user_id'), desc="召回评估"):
            with open(self.result_dir / f"{user_id}.json") as f:
                recalls = json.load(f)['recalls']

            true_items = group['adgroup_id'].unique()
            hits = len(set(recalls) & set(true_items))
            hit_counts.append(hits / len(true_items))

        recall = np.mean(hit_counts)
        print(f"\n召回评估 - Recall@{self.cfg.recall_top_k}: {recall:.4f}")
        return recall


# 排序模型
class RankingModel(tf.keras.Model):
    def __init__(self, user_dim, item_dim):
        super().__init__()
        # 特征交叉网络
        self.dense1 = Dense(256, activation='relu')
        self.bn1 = BatchNormalization()
        self.dense2 = Dense(128, activation='relu')
        self.bn2 = BatchNormalization()
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        # 合并用户和物品特征
        concat = Concatenate()([inputs['user_feat'], inputs['item_feat']])
        x = self.dense1(concat)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        return self.output_layer(x)


# 排序处理器
class RankingHandler:
    def __init__(self, config, processor):
        self.cfg = config
        self.processor = processor
        self.result_dir = config.result_dir / "ranking"
        os.makedirs(self.result_dir, exist_ok=True)

    def process(self, model):
        """执行排序评估"""
        results = []
        test_users = self.processor.test_data['user_id'].unique()

        for user_id in tqdm(test_users, desc="排序处理"):
            # 加载召回结果
            recall_path = self.cfg.result_dir / "recall" / f"{user_id}.json"
            if not recall_path.exists():
                continue

            with open(recall_path) as f:
                recall_data = json.load(f)

            # 获取用户特征
            user_data = self.processor.test_data[self.processor.test_data['user_id'] == user_id].iloc[0]
            user_feat = user_data[[c for c in self.processor.user_feature_cols if c != 'user_id']].values

            # 获取候选物品特征
            recall_items = recall_data['recalls']
            item_data = self.processor.test_data[
                self.processor.test_data['adgroup_id'].isin(recall_items)
            ].drop_duplicates('adgroup_id')

            if len(item_data) == 0:
                continue

            # 准备模型输入
            item_feats = item_data[[c for c in self.processor.item_feature_cols if c != 'adgroup_id']].values
            user_feats = np.tile(user_feat, (len(item_data), 1))

            # 预测
            predictions = model.predict(
                {
                    'user_feat': user_feats,
                    'item_feat': item_feats
                },
                verbose=0
            ).flatten()

            # 取Top2
            top2_idx = np.argsort(predictions)[-2:][::-1]
            top2_items = item_data.iloc[top2_idx]['adgroup_id'].values

            # 检查命中
            true_exposure = self.processor.test_data[
                (self.processor.test_data['user_id'] == user_id) &
                (self.processor.test_data['clk'] == 1)
                ]['adgroup_id'].values

            hits = [int(item in true_exposure) for item in top2_items]

            # 保存结果
            result = {
                "user_id": user_id,
                "top2_items": top2_items.tolist(),
                "hits": hits,
                "predictions": predictions[top2_idx].tolist()
            }
            results.append(result)

            with open(self.result_dir / f"{user_id}.json", 'w') as f:
                json.dump(result, f)

        # 计算全局指标
        total_hits = sum(sum(r['hits']) for r in results)
        total_possible = 2 * len(results)
        precision = total_hits / total_possible if total_possible > 0 else 0

        print(f"\n排序评估 - Precision@2: {precision:.4f}")
        return results


# 完整流程
class RecommenderSystem:
    def __init__(self, config):
        self.cfg = config
        self.processor = DataProcessor(config)
        self.recall_handler = None
        self.ranking_handler = None

    def run(self):
        # 1. 加载数据
        print("加载数据...")
        self.processor.load_all_data()

        # 2. 召回阶段
        print("\n=== 召回阶段 ===")
        recall_model = SequenceRecallModel(self.cfg)
        recall_model.compile(optimizer='adam', loss='mse')

        # 模拟训练（实际需要实现真实训练逻辑）
        print("训练召回模型...")
        # 这里应添加真实的训练代码

        self.recall_handler = RecallHandler(self.cfg, self.processor)
        self.recall_handler.generate_recalls(recall_model)
        self.recall_handler.evaluate_recall()

        # 3. 排序阶段
        print("\n=== 排序阶段 ===")
        ranking_model = RankingModel(
            user_dim=len([c for c in self.processor.user_feature_cols if c != 'user_id']),
            item_dim=len([c for c in self.processor.item_feature_cols if c != 'adgroup_id'])
        )
        ranking_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[AUC(name='auc')]
        )

        # 模拟训练（实际需要实现真实训练逻辑）
        print("训练排序模型...")
        # 这里应添加真实的训练代码

        self.ranking_handler = RankingHandler(self.cfg, self.processor)
        ranking_results = self.ranking_handler.process(ranking_model)

        # 保存最终结果
        self._save_final_results(ranking_results)

    def _save_final_results(self, results):
        """保存最终评估结果"""
        summary = {
            "recall_metric": self.recall_handler.evaluate_recall(),
            "precision_at_2": sum(sum(r['hits']) for r in results) / (2 * len(results))
        }

        with open(self.cfg.result_dir / "final_metrics.json", 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    # 初始化配置
    cfg = Config()

    # 创建结果目录
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.result_dir / "recall", exist_ok=True)
    os.makedirs(cfg.result_dir / "ranking", exist_ok=True)

    # 运行系统
    system = RecommenderSystem(cfg)
    system.run()