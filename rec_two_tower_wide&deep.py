import os
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss
import json
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Dot, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score, precision_score
from tensorflow.keras.metrics import AUC
import tensorflow_ranking as tfr
import matplotlib.pyplot as plt


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

    def load_data(self, train_path="data/processed_data3.parquet", test_path="data/processed_data_test3.parquet"):
        """加载数据并创建ID映射"""
        with tqdm(total=2, desc="加载数据") as pbar:
            self.train_data = pd.read_parquet(train_path)
            pbar.update(1)
            self.test_data = pd.read_parquet(test_path)
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


class UserProfileAccessor:
    """改进后的用户画像访问器，处理新用户情况"""

    def __init__(self, profile_path, hot_items_path=None):
        """
        初始化用户画像访问器
        :param profile_path: 用户画像数据路径
        :param hot_items_path: 热门物品数据路径(可选)
        """
        self.profile_path = profile_path
        self.hot_items = None
        if hot_items_path:
            self._load_hot_items(hot_items_path)
        self._init_data_source()

    def _load_hot_items(self, path):
        """加载热门物品数据"""
        if path.endswith('.parquet'):
            self.hot_items = pd.read_parquet(path)['adgroup_id'].values
        elif path.endswith('.npy'):
            self.hot_items = np.load(path)
        else:
            print(f"警告: 不支持的热门物品数据格式 {path}")

    def _init_data_source(self):
        """初始化数据源"""
        if self.profile_path.endswith('.parquet'):
            self.profile_df = pd.read_parquet(self.profile_path)
            self.profile_df.set_index('userid', inplace=True)
        else:
            raise ValueError("仅支持parquet文件")

    def get_user_features(self, user_id):
        """
        获取用户特征，处理新用户情况
        :param user_id: 用户ID
        :return: 用户特征字典或None(新用户)
        """
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
                'occupation': int(user_data['occupation']),
                'is_new_user': False  # 标记是否为新用户
            }
        except KeyError:
            print(f"信息: 用户 {user_id} 是新用户，将使用默认特征")
            return {
                'cms_segid': 0,  # 默认值
                'cms_group_id': 0,
                'final_gender_code': 0,
                'age_level': 0,
                'pvalue_level': 0,
                'shopping_level': 0,
                'new_user_class_level': 0,
                'occupation': 0,
                'is_new_user': True  # 标记为新用户
            }


class RecallSystem:
    """改进后的召回系统，处理新用户情况"""

    def __init__(self, user_tower_path, item_tower_path, item_index_path,
                 item_ids_path, user_profile_path, hot_items_path=None):
        """
        初始化召回系统
        :param hot_items_path: 热门物品数据路径(可选)
        """
        # 加载模型
        with tf.device('/GPU:0'):
            self.user_tower = tf.keras.models.load_model(user_tower_path)
            self.item_tower = tf.keras.models.load_model(item_tower_path)

        # 加载FAISS索引
        self.index = faiss.read_index(item_index_path)
        self.item_ids = np.load(item_ids_path)

        # 初始化用户画像访问器(带热门物品支持)
        self.user_profile = UserProfileAccessor(user_profile_path, hot_items_path)
        print(f"召回系统初始化完成，包含 {self.index.ntotal} 个物品向量")

    def recall_items(self, user_ids, top_k=100, fallback_to_hot=True):
        """
        改进后的召回方法，处理新用户情况
        :param user_ids: 用户ID列表
        :param top_k: 召回数量
        :param fallback_to_hot: 是否对新用户回退到热门物品
        :return: 召回结果字典 {user_id: [item_ids]}
        """
        results = {}

        # 分批处理用户
        for user_id in tqdm(user_ids, desc="召回处理"):
            features = self.user_profile.get_user_features(user_id)

            # 新用户处理逻辑
            if features['is_new_user'] and fallback_to_hot:
                if self.user_profile.hot_items is not None:
                    # 从热门物品中随机选择
                    hot_items = np.random.choice(
                        self.user_profile.hot_items,
                        size=min(top_k, len(self.user_profile.hot_items)),
                        replace=False
                    )
                    results[user_id] = hot_items
                    continue
                else:
                    print(f"警告: 无热门物品数据，无法为新用户 {user_id} 提供召回")

            # 正常用户处理
            inputs = {
                'user_cms_segid': np.array([features['cms_segid']], dtype=np.int32),
                'user_cms_group_id': np.array([features['cms_group_id']], dtype=np.int32),
                'user_gender': np.array([features['final_gender_code']], dtype=np.int32),
                'user_age_level': np.array([features['age_level']], dtype=np.int32),
                'user_pvalue_level': np.array([features['pvalue_level'] + 1], dtype=np.int32),
                'user_shopping_level': np.array([features['shopping_level']], dtype=np.int32),
                'user_city_level': np.array([features['new_user_class_level']], dtype=np.int32),
                'user_occupation': np.array([features['occupation']], dtype=np.float32)
            }

            # GPU预测
            with tf.device('/GPU:0'):
                user_vec = self.user_tower.predict(inputs)

            # FAISS搜索
            distances, indices = self.index.search(
                user_vec.astype('float32'),
                top_k
            )
            results[user_id] = self.item_ids[indices[0]]

        return results


class WideDeepRecommender:
    """使用Wide & Deep架构的推荐系统"""

    def __init__(self, num_users, num_items, user_feat_dim, item_feat_dim):
        """
        初始化
        :param num_users: 用户数量
        :param num_items: 物品数量
        :param user_feat_dim: 用户特征维度
        :param item_feat_dim: 物品特征维度
        """
        self.num_users = num_users
        self.num_items = num_items
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        self.model = self._build_model()

    def _build_model(self):
        """构建Wide & Deep模型"""
        # 1. 输入层
        user_id_input = Input(shape=(1,), name='user_id_input')
        item_id_input = Input(shape=(1,), name='item_id_input')
        user_feat_input = Input(shape=(self.user_feat_dim,), name='user_feat_input')
        item_feat_input = Input(shape=(self.item_feat_dim,), name='item_feat_input')

        # 2. Wide部分（记忆组件）
        # 用户ID和物品ID的交叉特征
        user_item_cross = tf.keras.layers.experimental.preprocessing.HashedCrossing(
            num_bins=self.num_users * self.num_items,
            output_mode='one_hot')([user_id_input, item_id_input])

        wide_output = Dense(1, activation='linear')(user_item_cross)

        # 3. Deep部分（泛化组件）
        # Embedding层
        user_embedding = Embedding(
            input_dim=self.num_users,
            output_dim=32,
            name='user_embedding')(user_id_input)
        user_embedding = tf.keras.layers.Flatten()(user_embedding)

        item_embedding = Embedding(
            input_dim=self.num_items,
            output_dim=32,
            name='item_embedding')(item_id_input)
        item_embedding = tf.keras.layers.Flatten()(item_embedding)

        # 特征处理
        user_feat_norm = BatchNormalization()(user_feat_input)
        item_feat_norm = BatchNormalization()(item_feat_input)

        # 合并所有Deep特征
        deep_features = Concatenate()([
            user_embedding, user_feat_norm,
            item_embedding, item_feat_norm
        ])

        # Deep神经网络
        deep_output = Dense(256, activation='relu')(deep_features)
        deep_output = BatchNormalization()(deep_output)
        deep_output = Dense(128, activation='relu')(deep_output)
        deep_output = BatchNormalization()(deep_output)
        deep_output = Dense(64, activation='relu')(deep_output)
        deep_output = Dense(1, activation='linear')(deep_output)

        # 4. 合并Wide和Deep部分
        combined_output = tf.keras.layers.add([wide_output, deep_output])
        output = tf.keras.layers.Activation('sigmoid')(combined_output)

        # 5. 构建模型
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
        """训练模型"""
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
        """评估模型"""
        y_pred = self.model.predict(X_test).flatten()
        return {
            'AUC': roc_auc_score(y_test, y_pred)
        }, y_pred


# 使用示例（替换原DeepFMRerank类）
class RecommenderSystem:
    """修改后的推荐系统（使用Wide & Deep）"""

    def __init__(self, config):
        """
        初始化推荐系统
        :param config: 配置字典，包含各种路径和参数
        """
        self.config = config

        # 初始化召回系统
        self.recall_system = RecallSystem(
            user_tower_path=config['user_tower_path'],
            item_tower_path=config['item_tower_path'],
            item_index_path=config['item_index_path'],
            item_ids_path=config['item_ids_path'],
            user_profile_path=config['user_profile_path'],
            hot_items_path=config.get('hot_items_path')
        )

        # 初始化数据处理
        self.data_processor = DataProcessor()
        self.ranking_model = None

        # 热门物品缓存
        self.hot_items_cache = None
        if config.get('hot_items_path'):
            self._load_hot_items(config['hot_items_path'])

    def _load_hot_items(self, path):
        """加载热门物品"""
        if path.endswith('.parquet'):
            df = pd.read_parquet(path)
            self.hot_items_cache = df['adgroup_id'].values
        elif path.endswith('.npy'):
            self.hot_items_cache = np.load(path)
        print(f"已加载 {len(self.hot_items_cache)} 个热门物品")

    def run(self):
        """运行完整推荐流程"""
        print("=" * 50)
        print("开始推荐系统完整流程(支持新用户)")
        print("=" * 50)

        # 1. 数据加载
        print("\n[阶段1] 数据加载与预处理...")
        self.data_processor.load_data()

        # 2. 训练排序模型
        print("\n[阶段2] 训练排序模型...")
        self._train_ranking_model()

        # 3. 评估流程
        print("\n[阶段3] 评估推荐系统...")
        self._evaluate_system()

        print("\n推荐系统流程完成!")

    def _evaluate_system(self):
        """评估系统(增加新用户处理)"""
        # 获取测试用户
        test_user_ids = self.data_processor.test_data['user_id'].unique()

        # 1. 召回阶段
        print("\n[召回阶段评估]")
        recalled_items = self.recall_system.recall_items(
            test_user_ids,
            top_k=100,
            fallback_to_hot=True
        )

        # 统计新用户比例
        new_user_count = sum(
            1 for user_id in test_user_ids
            if self.recall_system.user_profile.get_user_features(user_id)['is_new_user']
        )
        print(f"测试集中新用户比例: {new_user_count / len(test_user_ids):.2%}")

        # 2. 排序阶段
        print("\n[排序阶段评估]")
        X_test = [
            self.data_processor.test_data['user_id_encoded'].values.reshape(-1, 1),
            self.data_processor.test_data['item_id_encoded'].values.reshape(-1, 1),
            self.data_processor.test_data[
                [c for c in self.data_processor.user_feature_cols if c not in ['user_id']]].values,
            self.data_processor.test_data[
                [c for c in self.data_processor.item_feature_cols if c not in ['adgroup_id']]].values
        ]
        y_test = self.data_processor.test_data['clk'].values

        test_metrics, y_pred = self.ranking_model.evaluate(
            X_test,
            y_test,
            self.data_processor.test_data.copy()
        )

        print("\n测试集评估结果:")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")

        # 保存结果
        self._save_results(test_metrics, y_test, y_pred)

    def _save_results(self, metrics, y_true, y_pred):
        """保存结果"""
        os.makedirs(self.config['output_dir'], exist_ok=True)

        # 将 metrics 中的 numpy.float32 转换为 Python float
        serializable_metrics = {
            k: float(v) if isinstance(v, np.float32) else v
            for k, v in metrics.items()
        }

        # 保存指标
        with open(f"{self.config['output_dir']}/metrics.json", "w") as f:
            json.dump(serializable_metrics, f, indent=2)

        # 保存预测结果
        pd.DataFrame({
            'true': y_true,
            'pred': y_pred
        }).to_csv(f"{self.config['output_dir']}/predictions.csv", index=False)

    def _train_ranking_model(self):
        """训练排序模型（使用Wide & Deep）"""
        # 划分训练/验证集
        train_df, val_df = train_test_split(
            self.data_processor.train_data,
            test_size=0.2,
            random_state=42
        )

        # 初始化Wide & Deep模型
        self.ranking_model = WideDeepRecommender(
            num_users=self.data_processor.num_users,
            num_items=self.data_processor.num_items,
            user_feat_dim=len([c for c in self.data_processor.user_feature_cols if c not in ['user_id']]),
            item_feat_dim=len([c for c in self.data_processor.item_feature_cols if c not in ['adgroup_id']])
        )

        # 准备输入数据
        def prepare_inputs(df):
            return [
                df['user_id_encoded'].values.reshape(-1, 1),  # user_id_input
                df['item_id_encoded'].values.reshape(-1, 1),  # item_id_input
                df[[c for c in self.data_processor.user_feature_cols if c not in ['user_id']]].values,
                df[[c for c in self.data_processor.item_feature_cols if c not in ['adgroup_id']]].values
            ]

        X_train = prepare_inputs(train_df)
        y_train = train_df['clk'].values
        X_val = prepare_inputs(val_df)
        y_val = val_df['clk'].values

        # 训练模型
        print("\n训练Wide & Deep模型...")
        history = self.ranking_model.train(X_train, y_train, X_val, y_val)
        self._save_training_curves(history)

    def _save_training_curves(self, history):
        """保存训练曲线"""
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
        os.makedirs(self.config['output_dir'], exist_ok=True)
        plt.savefig(f"{self.config['output_dir']}/training_curve.png")
        plt.close()

    def recommend_for_new_user(self, user_id, top_k=10):
        """
        为新用户生成推荐
        :param user_id: 新用户ID
        :param top_k: 推荐数量
        :return: 推荐物品列表
        """
        if self.hot_items_cache is None:
            raise ValueError("未加载热门物品数据，无法为新用户推荐")

        # 从热门物品中随机选择
        recommended = np.random.choice(
            self.hot_items_cache,
            size=min(top_k, len(self.hot_items_cache)),
            replace=False
        )
        return recommended.tolist()



# 配置示例
config = {
    'user_tower_path': "results/0427embed/user_tower",
    'item_tower_path': "results/0427embed/item_tower",
    'item_index_path': "recall/item_index_1745730778.faiss",
    'item_ids_path': "recall/item_ids_1745730778.npy",
    'user_profile_path': "data/user.parquet",
    # 'hot_items_path': "data/hot_items.parquet",  # 新增热门物品数据路径
    'output_dir': "results/0429rec_tt_df/full_system_v2"
}

if __name__ == "__main__":
    # 初始化推荐系统
    recommender = RecommenderSystem(config)

    # 运行完整流程
    recommender.run()

    # 模拟新用户推荐
    new_user_id = "new_user_123"
    print(f"\n模拟新用户推荐({new_user_id}):")
    print(recommender.recommend_for_new_user(new_user_id, top_k=5))