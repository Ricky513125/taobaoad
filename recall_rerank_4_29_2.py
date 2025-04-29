import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Dot, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score, precision_score
from tensorflow.keras.metrics import AUC
import tensorflow_ranking as tfr
"""
4.29 13:12
将两个id输入到模型中进行训练
"""

# 设置GPU内存自动增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class DataProcessor:
    """数据处理管道（带ID Embedding版）"""

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.user_id_map = None
        self.item_id_map = None
        self.user_feature_cols = [
            'user_id', 'cms_segid', 'cms_group_id', 'final_gender_code',
            'age_level', 'pvalue_level', 'shopping_level',
            'occupation', 'new_user_class_level'
        ]
        self.item_feature_cols = [
            'adgroup_id', 'cate_id', 'campaign_id', 'brand', 'price'
        ]

    def load_data(self):
        """加载数据并创建ID映射"""
        with tqdm(total=2, desc="加载数据") as pbar:
            self.train_data = pd.read_parquet("data/processed_data3.parquet")
            pbar.update(1)
            self.test_data = pd.read_parquet("data/processed_data_test3.parquet")
            pbar.update(1)

        # 创建ID映射
        self._create_id_mappings()
        self._normalize_features()
        self._validate_columns()

    def _create_id_mappings(self):
        """创建用户和物品ID到连续整数的映射"""
        # 用户ID映射
        unique_users = pd.concat([
            self.train_data['user_id'],
            self.test_data['user_id']
        ]).unique()
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.num_users = len(self.user_id_map)

        # 物品ID映射
        unique_items = pd.concat([
            self.train_data['adgroup_id'],
            self.test_data['adgroup_id']
        ]).unique()
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.num_items = len(self.item_id_map)

        # 应用映射
        self.train_data['user_id_encoded'] = self.train_data['user_id'].map(self.user_id_map)
        self.test_data['user_id_encoded'] = self.test_data['user_id'].map(self.user_id_map)
        self.train_data['item_id_encoded'] = self.train_data['adgroup_id'].map(self.item_id_map)
        self.test_data['item_id_encoded'] = self.test_data['adgroup_id'].map(self.item_id_map)

    def _normalize_features(self):
        """数值特征标准化"""
        numeric_cols = ['price', 'pvalue_level', 'shopping_level']
        for col in numeric_cols:
            if col in self.train_data.columns:
                mean = self.train_data[col].mean()
                std = self.train_data[col].std()
                self.train_data[col] = (self.train_data[col] - mean) / std
                self.test_data[col] = (self.test_data[col] - mean) / std

    def _validate_columns(self):
        """验证列名"""
        required_cols = set(self.user_feature_cols + self.item_feature_cols + ['clk'])
        missing_cols = required_cols - set(self.train_data.columns)
        if missing_cols:
            raise ValueError(f"缺失列: {missing_cols}")


class DeepFMRerank:
    """DeepFM模型（带ID Embedding版）"""

    def __init__(self, num_users, num_items, user_feat_dim, item_feat_dim):
        self.num_users = num_users
        self.num_items = num_items
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        self.model = self._build_model()
        print(f"\n初始化模型: 用户数={num_users}, 物品数={num_items}")
        print(f"用户特征维度={user_feat_dim}, 物品特征维度={item_feat_dim}")

    def _build_model(self):
        """构建带ID Embedding的DeepFM模型"""
        # 1. 输入层
        user_id_input = Input(shape=(1,), name='user_id_input')
        item_id_input = Input(shape=(1,), name='item_id_input')
        user_feat_input = Input(shape=(self.user_feat_dim,), name='user_feat_input')
        item_feat_input = Input(shape=(self.item_feat_dim,), name='item_feat_input')

        # 2. ID Embedding层
        user_embedding = Embedding(
            input_dim=self.num_users,
            output_dim=32,
            name='user_embedding'
        )(user_id_input)
        user_embedding = Flatten()(user_embedding)

        item_embedding = Embedding(
            input_dim=self.num_items,
            output_dim=32,
            name='item_embedding'
        )(item_id_input)
        item_embedding = Flatten()(item_embedding)

        # 3. 特征标准化
        user_feat_norm = BatchNormalization()(user_feat_input)
        item_feat_norm = BatchNormalization()(item_feat_input)

        # 4. FM部分
        # 用户侧
        user_feat_emb = Dense(64, activation='relu')(user_feat_norm)
        user_combined = Concatenate()([user_embedding, user_feat_emb])
        user_fm = Dense(64, activation='relu')(user_combined)

        # 物品侧
        item_feat_emb = Dense(64, activation='relu')(item_feat_norm)
        item_combined = Concatenate()([item_embedding, item_feat_emb])
        item_fm = Dense(64, activation='relu')(item_combined)

        # FM交互
        fm_output = Dot(axes=1)([user_fm, item_fm])

        # 5. DNN部分
        concat_features = Concatenate()([
            user_embedding, user_feat_input,
            item_embedding, item_feat_input
        ])
        dnn_out = Dense(256, activation='relu')(concat_features)
        dnn_out = BatchNormalization()(dnn_out)
        dnn_out = Dense(128, activation='relu')(dnn_out)
        dnn_out = BatchNormalization()(dnn_out)

        # 6. 输出层
        output = Dense(1, activation='sigmoid', name='ctr')(
            Concatenate()([fm_output, dnn_out])
        )

        # 7. 构建模型
        model = Model(
            inputs=[user_id_input, item_id_input, user_feat_input, item_feat_input],
            outputs=output
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=[AUC(name='auc')]
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

    def evaluate(self, X_test, y_test, test_df):
        """评估方法"""
        # 生成预测
        y_pred = self.model.predict(X_test, batch_size=4096).flatten()
        test_df['pred'] = y_pred

        # 初始化NDCG指标
        ndcg_metric = tfr.keras.metrics.NDCGMetric(name='ndcg_2', topn=2)

        # 按用户分组评估
        user_groups = test_df.groupby('user_id')
        ndcg_scores = []
        precision_scores = []

        for user_id, group in user_groups:
            if len(group) < 2:  # 跳过交互不足的用户
                continue

            pred = group['pred'].values
            true = group['clk'].values

            # 计算NDCG
            ndcg_metric.update_state([true], [pred])
            ndcg = ndcg_metric.result().numpy()
            ndcg_scores.append(ndcg)
            ndcg_metric.reset_states()

            # 计算Precision@2
            top_k_idx = np.argsort(pred)[-2:]
            precision_scores.append(true[top_k_idx].sum() / 2)

        metrics = {
            'AUC': roc_auc_score(y_test, y_pred),
            'NDCG@2': np.mean(ndcg_scores) if ndcg_scores else 0,
            'Precision@2': np.mean(precision_scores) if precision_scores else 0
        }
        return metrics, y_pred


class Trainer:
    """训练管道（带ID Embedding版）"""

    def __init__(self):
        self.processor = DataProcessor()
        self.model = None

    def run(self):
        # 1. 加载数据
        self.processor.load_data()

        # 2. 划分训练/验证集
        train_df, val_df = train_test_split(
            self.processor.train_data,
            test_size=0.2,
            random_state=42
        )

        # 3. 初始化模型
        self.model = DeepFMRerank(
            num_users=self.processor.num_users,
            num_items=self.processor.num_items,
            user_feat_dim=len([c for c in self.processor.user_feature_cols if c not in ['user_id']]),
            item_feat_dim=len([c for c in self.processor.item_feature_cols if c not in ['adgroup_id']])
        )

        # 4. 准备输入数据
        def prepare_inputs(df):
            return [
                df['user_id_encoded'].values.reshape(-1, 1),  # user_id_input
                df['item_id_encoded'].values.reshape(-1, 1),  # item_id_input
                df[[c for c in self.processor.user_feature_cols if c not in ['user_id']]].values,  # user_feat_input
                df[[c for c in self.processor.item_feature_cols if c not in ['adgroup_id']]].values  # item_feat_input
            ]

        X_train = prepare_inputs(train_df)
        y_train = train_df['clk'].values
        X_val = prepare_inputs(val_df)
        y_val = val_df['clk'].values

        # 5. 训练
        print("\n训练数据统计:")
        print(f"用户ID范围: {train_df['user_id_encoded'].min()}~{train_df['user_id_encoded'].max()}")
        print(f"物品ID范围: {train_df['item_id_encoded'].min()}~{train_df['item_id_encoded'].max()}")
        print(f"正样本比例: {y_train.mean():.2%}")

        history = self.model.train(X_train, y_train, X_val, y_val)

        # 6. 测试集评估
        X_test = prepare_inputs(self.processor.test_data)
        y_test = self.processor.test_data['clk'].values

        test_metrics, y_pred = self.model.evaluate(
            X_test,
            y_test,
            self.processor.test_data.copy()
        )

        print("\n测试集评估结果:")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")

        # 7. 保存结果
        self.save_results(history, test_metrics, y_test, y_pred)

    def save_results(self, history, metrics, y_true, y_pred):
        """保存结果"""
        os.makedirs("results/recall_with_ids_4_29", exist_ok=True)

        # 保存指标
        with open("results/recall_with_ids_4_29/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # 保存预测结果
        pd.DataFrame({
            'true': y_true,
            'pred': y_pred
        }).to_csv("results/recall_with_ids_4_29/predictions.csv", index=False)

        # 绘制训练曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Val AUC')
        plt.title('Model AUC')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('results/recall_with_ids_4_29/training_curve.png')
        plt.close()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()