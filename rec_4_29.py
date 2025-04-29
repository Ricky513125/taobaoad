"""
推荐系统全流程实现（召回+排序）
文件结构：
project/
├── data/
│   ├── processed/
│   │   ├── train.parquet
│   │   └── test.parquet
│   ├── sequences/
│   │   ├── user_sequences.csv
│   │   └── item_graph.pkl
├── models/
│   ├── recall_model/
│   └── ranking_model/
├── results/
│   ├── recall/
│   └── ranking/
└── main.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from pathlib import Path
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate
from tensorflow.keras.models import Model

"""
4.29 
利用序列数据的完整模型评估

"""
# 配置参数
class Config:
    # 路径配置
    data_dir = Path("data/processed")
    seq_dir = Path("data/sequences")
    model_dir = Path("models")
    result_dir = Path("results")

    # 召回参数
    recall_top_k = 100
    user_seq_len = 20  # 用户行为序列长度

    # 模型参数
    user_embed_dim = 64
    item_embed_dim = 64
    lstm_units = 32

    # 训练参数
    batch_size = 4096
    epochs = 10


# 数据处理器（新增序列处理）
class SequenceProcessor:
    def __init__(self, config):
        self.cfg = config

    def load_user_sequences(self):
        """加载用户行为序列"""
        seq_path = self.cfg.seq_dir / "user_sequences.csv"
        df = pd.read_csv(seq_path)

        # 转换为序列字典 {user_id: [item1, item2, ...]}
        seq_dict = df.groupby('user_id')['item_id'].apply(list).to_dict()
        return seq_dict

    def load_item_graph_embeddings(self):
        """加载物品图嵌入"""
        graph_path = self.cfg.seq_dir / "item_graph.pkl"
        with open(graph_path, 'rb') as f:
            return pickle.load(f)  # 假设格式为{item_id: embedding}


# 召回模型（整合序列特征）
class SequenceRecallModel(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 用户塔
        self.user_encoder = self._build_user_tower()

        # 物品塔
        self.item_encoder = self._build_item_tower()

    def _build_user_tower(self):
        """用户特征编码器（含序列）"""
        # 基础特征
        base_input = Input(shape=(8,))  # 假设8个用户特征

        # 行为序列特征
        seq_input = Input(shape=(self.cfg.user_seq_len,))
        seq_emb = Embedding(10000, 32)(seq_input)  # 假设物品ID总数1w
        seq_feat = LSTM(self.cfg.lstm_units)(seq_emb)

        # 合并特征
        concat = Concatenate()([base_input, seq_feat])
        output = Dense(self.cfg.user_embed_dim)(concat)
        return Model(inputs=[base_input, seq_input], outputs=output)

    def _build_item_tower(self):
        """物品特征编码器（含图嵌入）"""
        # 基础特征
        base_input = Input(shape=(4,))  # 假设4个物品特征

        # 图嵌入特征
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
    def __init__(self, cfg):
        self.cfg = cfg
        self.result_dir = cfg.result_dir / "recall"
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def generate_recalls(self, model, users, items):
        """生成召回结果并保存"""
        # 生成物品嵌入
        item_embs = model.item_encoder.predict(
            {'item_base': items[cfg.item_base_cols].values,
             'item_graph': items['graph_emb'].values},
            batch_size=self.cfg.batch_size
        )

        # 为每个用户生成召回
        for user_id, user_data in tqdm(users.iterrows(), desc="生成召回"):
            user_emb = model.user_encoder.predict({
                'user_base': user_data[cfg.user_base_cols].values.reshape(1, -1),
                'user_seq': pad_sequence(user_data['hist_seq'])
            })

            # 计算相似度（简化示例）
            scores = np.dot(user_emb, item_embs.T)
            top_k = np.argsort(scores)[-self.cfg.recall_top_k:]

            # 保存结果
            with open(self.result_dir / f"{user_id}.json", 'w') as f:
                json.dump({
                    "user_id": int(user_id),
                    "recalls": items.iloc[top_k]['item_id'].tolist()
                }, f)

    def evaluate_recall(self, test_data):
        """召回评估"""
        hit_counts = []
        for user_id, true_items in test_data.groupby('user_id')['item_id']:
            with open(self.result_dir / f"{user_id}.json") as f:
                recalls = json.load(f)['recalls']
            hits = len(set(recalls) & set(true_items))
            hit_counts.append(hits / len(true_items))

        recall = np.mean(hit_counts)
        print(f"Recall@{self.cfg.recall_top_k}: {recall:.4f}")
        return recall


# 排序模型（适配召回结果）
class RankingModel(tf.keras.Model):
    def __init__(self, user_dim, item_dim):
        super().__init__()
        # 特征交叉层
        self.cross_layer = Dense(256, activation='relu')
        # 预测层
        self.pred_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        concat = Concatenate()([inputs['user_feat'], inputs['item_feat']])
        cross = self.cross_layer(concat)
        return self.pred_layer(cross)


# 排序处理器
class RankingHandler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.result_dir = cfg.result_dir / "ranking"
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def process(self, model, test_data):
        """处理排序并评估"""
        results = []
        for user_id in tqdm(test_data['user_id'].unique(), desc="排序处理"):
            # 加载召回结果
            with open(self.cfg.result_dir / "recall" / f"{user_id}.json") as f:
                recalls = json.load(f)['recalls'][:100]  # 取前100

            # 获取真实曝光
            true_exposure = test_data[test_data['user_id'] == user_id]
            true_items = true_exposure['item_id'].values

            # 生成预测
            user_feat = get_user_features(user_id)
            item_feats = get_item_features(recalls)
            preds = model.predict({
                'user_feat': np.tile(user_feat, (len(recalls), 1)),
                'item_feat': item_feats
            }).flatten()

            # 取Top2
            top2_idx = np.argsort(preds)[-2:][::-1]
            top2_items = [recalls[i] for i in top2_idx]

            # 保存结果
            result = {
                "user_id": user_id,
                "top2": top2_items,
                "hits": [int(item in true_items) for item in top2_items]
            }
            results.append(result)

            with open(self.result_dir / f"{user_id}.json", 'w') as f:
                json.dump(result, f)

        # 计算全局指标
        total_hits = sum(sum(r['hits']) for r in results)
        total_possible = 2 * len(results)
        precision = total_hits / total_possible
        print(f"Precision@2: {precision:.4f}")
        return results


# 完整流程
class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = None
        self.seq_processor = SequenceProcessor(cfg)

    def run(self):
        # 1. 加载数据
        train_data = pd.read_parquet(cfg.data_dir / 'train.parquet')
        test_data = pd.read_parquet(cfg.data_dir / 'test.parquet')

        # 2. 加载序列数据
        user_sequences = self.seq_processor.load_user_sequences()
        item_graph = self.seq_processor.load_item_graph_embeddings()

        # 3. 训练召回模型
        recall_model = self.train_recall(train_data, user_sequences, item_graph)

        # 4. 生成召回结果
        recall_handler = RecallHandler(cfg)
        recall_handler.generate_recalls(recall_model, train_data, item_graph)
        recall_handler.evaluate_recall(test_data)

        # 5. 训练排序模型
        rank_model = self.train_ranking(train_data)

        # 6. 排序处理
        ranking_handler = RankingHandler(cfg)
        ranking_handler.process(rank_model, test_data)

    def train_recall(self, data, user_sequences, item_graph):
        """训练召回模型（示例）"""
        # 需要根据实际数据构造训练样本
        model = SequenceRecallModel(self.cfg)
        model.compile(optimizer='adam', loss='mse')
        # 模拟训练
        model.fit(x=dummy_data, epochs=self.cfg.epochs)
        return model

    def train_ranking(self, data):
        """训练排序模型（示例）"""
        model = RankingModel(user_dim=64, item_dim=64)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        # 模拟训练
        model.fit(x=dummy_data, epochs=self.cfg.epochs)
        return model


if __name__ == "__main__":
    cfg = Config()
    pipeline = Pipeline(cfg)
    pipeline.run()