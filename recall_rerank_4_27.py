import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json
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
    """简化版数据处理管道（使用预合并数据）"""

    def __init__(self):
        self.train_data = None
        self.test_data = None
        # 明确定义特征列（需与合并后的列名一致）
        self.user_feature_cols = [
            'cms_segid', 'cms_group_id', 'final_gender_code',
            'age_level', 'pvalue_level', 'shopping_level',
            'occupation', 'new_user_class_level'
        ]
        self.item_feature_cols = [
            'cate_id', 'campaign_id', 'brand', 'price'
        ]

    def load_data(self):
        """直接加载预合并的数据"""
        with tqdm(total=2, desc="加载数据") as pbar:
            self.train_data = pd.read_parquet("data/combined_train.parquet")
            pbar.update(1)
            self.test_data = pd.read_parquet("data/combined_test.parquet")
            pbar.update(1)
        print(self.train_data.shape)
        print(self.test_data.shape)
        print(self.train_data.columns)
        print(self.test_data.columns)
        # 验证列名
        self._validate_columns()

    def _validate_columns(self):
        """验证合并后的数据包含所有需要的列"""
        required_cols = set(self.user_feature_cols + self.item_feature_cols + ['clk'])
        missing_cols = required_cols - set(self.train_data.columns)
        if missing_cols:
            raise ValueError(f"合并后的数据缺失以下列: {missing_cols}")


class DeepFMRerank:
    """DeepFM精排模型（修正版）"""

    def __init__(self, user_feat_dim, item_feat_dim):
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        self.model = self._build_model()
        print(f"\n初始化模型: 用户特征维度={user_feat_dim}, 物品特征维度={item_feat_dim}")

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
            epochs=100,
            batch_size=4096,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test):
        """模型评估"""
        y_pred = self.model.predict(X_test, batch_size=4096).flatten()

        metrics = {
            'AUC': roc_auc_score(y_test, y_pred),
            'NDCG@10': self.calculate_ndcg(y_test, y_pred, k=10),
            'Precision@10': precision_score(y_test, (y_pred > 0.5).astype(int), zero_division=0)
        }
        return metrics, y_pred

    def calculate_ndcg(self, y_true, y_pred, k=10):
        top_k_idx = np.argsort(y_pred)[-k:]
        sorted_labels = y_true[top_k_idx]
        ideal_labels = np.sort(y_true)[-k:]
        return ndcg_score([ideal_labels], [sorted_labels])


class Trainer:
    """训练评估管道（修正版）"""

    def __init__(self):
        self.processor = DataProcessor()
        self.model = None

    def run(self):
        # 1. 加载预合并数据
        self.processor.load_data()

        # 2. 划分训练/验证集
        train_df, val_df = train_test_split(
            self.processor.train_data,
            test_size=0.2,
            random_state=42
        )

        # 3. 初始化模型
        self.model = DeepFMRerank(
            user_feat_dim=len(self.processor.user_feature_cols),
            item_feat_dim=len(self.processor.item_feature_cols)
        )

        # 4. 准备输入数据
        X_train = [
            train_df[self.processor.user_feature_cols].values,
            train_df[self.processor.item_feature_cols].values
        ]
        y_train = train_df['clk'].values

        X_val = [
            val_df[self.processor.user_feature_cols].values,
            val_df[self.processor.item_feature_cols].values
        ]
        y_val = val_df['clk'].values

        # 5. 训练和评估
        print("\n训练数据统计:")
        print(f"用户特征维度: {X_train[0].shape}")
        print(f"物品特征维度: {X_train[1].shape}")
        print(f"正样本比例: {y_train.mean():.2%}")

        history = self.model.train(X_train, y_train, X_val, y_val)

        # 6. 测试集评估
        X_test = [
            self.processor.test_data[self.processor.user_feature_cols].values,
            self.processor.test_data[self.processor.item_feature_cols].values
        ]
        y_test = self.processor.test_data['clk'].values

        test_metrics, y_pred = self.model.evaluate(X_test, y_test)
        print("\n测试集评估结果:")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")

        # 7. 保存结果
        self.save_results(history, test_metrics, y_test, y_pred)

    def save_results(self, history, metrics, y_true, y_pred):
        """保存评估结果"""
        os.makedirs("results", exist_ok=True)

        # 保存指标
        with open("results/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # 保存预测结果
        pd.DataFrame({
            'true': y_true,
            'pred': y_pred
        }).to_csv("results/predictions.csv", index=False)

        # 绘制训练曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['AUC'], label='Train AUC')
        plt.plot(history.history['val_AUC'], label='Val AUC')
        plt.title('Model AUC')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('results/training_curve.png')
        plt.close()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()