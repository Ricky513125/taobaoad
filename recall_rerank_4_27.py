import os
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Dot, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score, ndcg_score, precision_score

# 设置GPU内存自动增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class DataProcessor:
    """数据处理管道"""

    def __init__(self):
        self.user_features = None
        self.item_features = None
        self.train_data = None
        self.test_data = None

    def load_data(self):
        """加载所有原始数据"""
        with tqdm(total=4, desc="加载数据") as pbar:
            # 用户特征
            self.user_features = pd.read_parquet("data/user.parquet")
            pbar.update(1)

            # 广告特征
            self.item_features = pd.read_csv("data/ad_feature.csv")
            self.item_features.rename(columns={'adgroup_id': 'item_id'}, inplace=True)
            pbar.update(1)

            # 训练数据
            self.train_data = pd.read_parquet("data/processed_data_train.parquet")
            pbar.update(1)

            # 测试数据
            self.test_data = pd.read_parquet("data/processed_data_test.parquet")
            pbar.update(1)

    def preprocess(self):
        """数据预处理"""
        with tqdm(total=3, desc="数据预处理") as pbar:
            # 用户特征处理
            self.user_features['user_id'] = self.user_features['userid'].astype(int)
            self.user_features.set_index('userid', inplace=True)
            pbar.update(1)

            # 物品特征处理
            self.item_features['item_id'] = self.item_features['item_id'].astype(int)
            self.item_features.set_index('item_id', inplace=True)
            pbar.update(1)

            # 合并点击日志
            self.train_data = self.train_data.merge(
                self.user_features, left_on='user', right_index=True, how='left'
            ).merge(
                self.item_features, left_on='adgroup_id', right_index=True, how='left'
            )
            pbar.update(1)


class DeepFMRerank:
    """DeepFM精排模型"""

    def __init__(self, user_feat_dim, item_feat_dim):
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        self.model = self._build_model()

    def _build_model(self):
        """构建DeepFM模型"""
        # 输入层
        user_input = Input(shape=(self.user_feat_dim,), name='user_input')
        item_input = Input(shape=(self.item_feat_dim,), name='item_input')

        # FM部分
        user_emb = Dense(64, activation='relu')(user_input)
        item_emb = Dense(64, activation='relu')(item_input)
        fm_output = Dot(axes=1)([user_emb, item_emb])

        # DNN部分
        concat = Concatenate()([user_input, item_input])
        dnn_out = Dense(256, activation='relu')(concat)
        dnn_out = BatchNormalization()(dnn_out)
        dnn_out = Dense(128, activation='relu')(dnn_out)
        dnn_out = BatchNormalization()(dnn_out)

        # 输出层
        output = Dense(1, activation='sigmoid', name='ctr')(
            Concatenate()([fm_output, dnn_out])
        )

        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=['AUC']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val):
        """模型训练"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=4096,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test):
        """模型评估"""
        # 预测CTR
        y_pred = self.model.predict(X_test, batch_size=4096).flatten()

        # 计算指标
        metrics = {
            'AUC': roc_auc_score(y_test, y_pred),
            'NDCG@10': self.calculate_ndcg(y_test, y_pred, k=10),
            'Precision@10': precision_score(y_test, (y_pred > 0.5).astype(int))
        }
        return metrics, y_pred

    def calculate_ndcg(self, y_true, y_pred, k=10):
        """计算NDCG指标"""
        top_k_idx = np.argsort(y_pred)[-k:]
        sorted_labels = y_true[top_k_idx]
        ideal_labels = np.sort(y_true)[-k:]
        return ndcg_score([ideal_labels], [sorted_labels])


class Trainer:
    """训练评估管道"""

    def __init__(self):
        self.processor = DataProcessor()
        self.model = None

    def run(self):
        """完整训练评估流程"""
        # 1. 数据准备
        self.processor.load_data()
        self.processor.preprocess()

        # 2. 准备训练数据
        train_df, val_df = train_test_split(
            self.processor.train_data,
            test_size=0.2,
            random_state=42
        )

        # 3. 训练模型
        self.model = DeepFMRerank(
            user_feat_dim=len(self.processor.user_features.columns),
            item_feat_dim=len(self.processor.item_features.columns)
        )

        X_train = [
            train_df[self.processor.user_features.columns].values,
            train_df[self.processor.item_features.columns].values
        ]
        y_train = train_df['clk'].values

        X_val = [
            val_df[self.processor.user_features.columns].values,
            val_df[self.processor.item_features.columns].values
        ]
        y_val = val_df['clk'].values

        history = self.model.train(X_train, y_train, X_val, y_val)

        # 4. 验证集评估
        val_metrics, _ = self.model.evaluate(X_val, y_val)
        print("\n验证集评估结果:")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")

        # 5. 测试集评估
        X_test = [
            self.processor.test_data[self.processor.user_features.columns].values,
            self.processor.test_data[self.processor.item_features.columns].values
        ]
        y_test = self.processor.test_data['clk'].values

        test_metrics, y_pred = self.model.evaluate(X_test, y_test)
        print("\n测试集评估结果:")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")

        # 6. 可视化结果
        self.plot_results(history, y_test, y_pred)

    def plot_results(self, history, y_true, y_pred):
        """可视化评估结果"""
        plt.figure(figsize=(15, 5))

        # 训练曲线
        plt.subplot(1, 3, 1)
        plt.plot(history.history['AUC'], label='Train AUC')
        plt.plot(history.history['val_AUC'], label='Val AUC')
        plt.title('Model AUC')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend()

        # CTR分布
        plt.subplot(1, 3, 2)
        plt.hist(y_pred[y_true == 1], bins=30, alpha=0.5, label='Positive')
        plt.hist(y_pred[y_true == 0], bins=30, alpha=0.5, label='Negative')
        plt.title('CTR Distribution')
        plt.legend()

        # 混淆矩阵
        plt.subplot(1, 3, 3)
        conf_matrix = tf.math.confusion_matrix(
            y_true, (y_pred > 0.5).astype(int)
        ).numpy()
        plt.imshow(conf_matrix, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig('results/evaluation_metrics.png')
        plt.close()


if __name__ == "__main__":
    # 创建结果目录
    os.makedirs("results", exist_ok=True)

    # 运行训练评估流程
    trainer = Trainer()
    trainer.run()