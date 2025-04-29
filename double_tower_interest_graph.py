# complete_double_tower_with_graph.py
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict
from sklearn.preprocessing import normalize
from node2vec import Node2Vec
import networkx as nx
from tqdm import tqdm

# GPU配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class ItemGraphGenerator:
    """物料关系图处理器"""

    def __init__(self, cooccur_threshold=5):
        self.threshold = cooccur_threshold
        self.graph = nx.DiGraph()

    def build_graph(self, df):
        """从行为数据构建有向关系图"""
        df['node'] = list(zip(df['cate'], df['brand']))

        # 时间衰减权重
        latest_time = df['time_stamp'].max()
        df['time_weight'] = 0.9 ** ((latest_time - df['time_stamp']) / 86400)

        # 构建共现关系
        cooccur = defaultdict(lambda: defaultdict(float))
        for _, group in df.groupby('user_id'):
            sorted_actions = group.sort_values('time_stamp')
            nodes = sorted_actions['node'].values
            weights = sorted_actions['time_weight'].values

            for i in range(len(nodes) - 1):
                src, dst = nodes[i], nodes[i + 1]
                cooccur[src][dst] += weights[i] * weights[i + 1]

        # 添加边到图
        for src, neighbors in cooccur.items():
            for dst, weight in neighbors.items():
                if weight >= self.threshold:
                    self.graph.add_edge(src, dst, weight=weight)
        return self.graph

    def generate_sequences(self, walk_length=20, num_walks=10):
        """生成随机游走序列"""
        node2vec = Node2Vec(
            self.graph,
            dimensions=64,
            walk_length=walk_length,
            num_walks=num_walks,
            p=1.0,
            q=0.5,
            workers=os.cpu_count()
        )
        return node2vec.walks


class EnhancedDoubleTower:
    """增强版双塔模型（整合关系图）"""

    def __init__(self, user_tower, item_tower, embed_size=64):
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.embed_size = embed_size
        self.item_vectors = None
        self.user_vectors = None

    def train_graph_embeddings(self, behavior_df):
        """训练图增强的物品向量"""
        # 1. 构建关系图
        graph_builder = ItemGraphGenerator()
        item_graph = graph_builder.build_graph(behavior_df)

        # 2. 生成随机游走序列
        sequences = graph_builder.generate_sequences()

        # 3. 训练物品向量
        from gensim.models import Word2Vec
        model = Word2Vec(
            sentences=sequences,
            vector_size=self.embed_size,
            window=5,
            min_count=3,
            workers=os.cpu_count()
        )

        # 4. 保存物品向量
        self.item_vectors = {
            node: model.wv[node]
            for node in model.wv.index_to_key
        }
        return self.item_vectors

    def build_joint_model(self):
        """构建联合模型架构"""
        # 原始双塔输入
        user_inputs = {k: v for k, v in self.user_tower.input.items()}
        item_inputs = {k: v for k, v in self.item_tower.input.items()}

        # 原始塔输出
        user_embed = self.user_tower(user_inputs)
        item_embed = self.item_tower(item_inputs)

        # 图增强物品向量输入
        item_graph_input = tf.keras.Input(shape=(self.embed_size,), name='item_graph_input')

        # 联合物品表征
        item_joint = tf.keras.layers.Concatenate()([
            item_embed,
            tf.keras.layers.Dense(self.embed_size)(item_graph_input)
        ])

        # 相似度计算
        dot_product = tf.keras.layers.Dot(axes=1, normalize=True)([user_embed, item_joint])
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_product)

        return tf.keras.Model(
            inputs={**user_inputs, ** item_inputs, 'item_graph_input': item_graph_input},
        outputs = output
        )

    def create_enhanced_dataset(self, data_path, batch_size=1024):
        """创建包含图特征的数据管道"""
        data = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)

        # 原始特征
        user_features = {
            'user_gender': data['final_gender_code'].values,
            'user_cms_segid': data['cms_segid'].values,
            'user_cms_group_id': data['cms_group_id'].values,
            'user_age_level': data['age_level'].values,
            'user_pvalue_level': data['pvalue_level'].values + 1,
            'user_shopping_level': data['shopping_level'].values,
            'user_city_level': data['new_user_class_level'].values,
            'user_occupation': data['occupation'].values.astype('float32')
        }
        item_features = {
            'item_adgroup_id': data['adgroup_id'].values,
            'item_cate_id': data['cate_id'].values,
            'item_campaign_id': data['campaign_id'].values.astype('int32'),
            'item_customer': data['customer'].values,
            'item_brand': data['brand'].values.astype('int32'),
            'item_price': data['price'].values.astype('float32')
        }

        # 添加图特征
        item_nodes = list(zip(data['cate_id'], data['brand']))
        graph_features = {
            'item_graph_input': np.array([
                self.item_vectors.get(node, np.zeros(self.embed_size))
                for node in item_nodes
            ])
        }

        labels = data['clk'].values

        return tf.data.Dataset.from_tensor_slices((
            {**user_features, **item_features, **graph_features},
        labels
        )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_existing_model(model_dir):
    """加载预训练双塔模型"""
    user_tower = tf.keras.models.load_model(f'{model_dir}/user_tower')
    item_tower = tf.keras.models.load_model(f'{model_dir}/item_tower')
    return user_tower, item_tower

def main():
    # 1. 加载现有模型
    user_tower, item_tower = load_existing_model('results/0427embed')

    # 2. 初始化增强系统
    enhanced_model = EnhancedDoubleTower(user_tower, item_tower)

    # 3. 训练图嵌入
    behavior_data = pd.read_csv('data/cleaned_behavior.csv')
    enhanced_model.train_graph_embeddings(behavior_data)

    # 4. 构建联合模型
    joint_model = enhanced_model.build_joint_model()
    joint_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['AUC']
    )

    # 5. 准备数据
    train_ds = enhanced_model.create_enhanced_dataset('data/processed_data3.parquet')
    test_ds = enhanced_model.create_enhanced_dataset('data/processed_data_test3.parquet')

    # 6. 训练配置
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/double_tower_interest_graph/enhanced_model_{epoch}.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(patience=3)
    ]

    # 7. 微调训练
    history = joint_model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=20,
        callbacks=callbacks
    )

    # 8. 保存最终模型
    joint_model.save('results/double_tower_interest_graph/enhanced_final_model')
    print("Model saved with graph enhancement!")

if __name__ == "__main__":
    main()