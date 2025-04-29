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
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time
from datetime import datetime

# GPU配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class ProgressVisualizer:
    """训练过程可视化工具"""

    def __init__(self):
        self.metrics = {
            'loss': [],
            'auc': [],
            'val_loss': [],
            'val_auc': []
        }
        self.start_time = None
        self.epoch_times = []

    def start_timer(self):
        self.start_time = time.time()

    def record_metrics(self, history, epoch):
        for k in self.metrics.keys():
            if k in history.history:
                self.metrics[k].append(history.history[k][0])

        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        avg_time = np.mean(self.epoch_times)
        remaining = avg_time * (len(self.epoch_times) - epoch - 1)

        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"Estimated remaining time: {remaining / 60:.1f} minutes")
        self.start_timer()

    def plot_metrics(self):
        plt.figure(figsize=(12, 5))

        # Loss曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['loss'], label='Train Loss')
        plt.plot(self.metrics['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # AUC曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['auc'], label='Train AUC')
        plt.plot(self.metrics['val_auc'], label='Validation AUC')
        plt.title('Training and Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()


class ItemGraphGenerator:
    """物料关系图处理器"""

    def __init__(self, cooccur_threshold=5):
        self.threshold = cooccur_threshold
        self.graph = nx.DiGraph()

    def build_graph(self, df):
        """从行为数据构建有向关系图"""
        print("\n🛠️ Building item relationship graph...")
        df['node'] = list(zip(df['cate'], df['brand']))

        # 时间衰减权重
        latest_time = df['time_stamp'].max()
        df['time_weight'] = 0.9 **((latest_time - df['time_stamp']) / 86400)

        # 构建共现关系
        cooccur = defaultdict(lambda: defaultdict(float))

        # 使用tqdm显示进度
        for _, group in tqdm(df.groupby('user_id'), desc="Processing user behaviors"):
            sorted_actions = group.sort_values('time_stamp')
            nodes = sorted_actions['node'].values
            weights = sorted_actions['time_weight'].values

            for i in range(len(nodes) - 1):
                src, dst = nodes[i], nodes[i + 1]
                cooccur[src][dst] += weights[i] * weights[i + 1]

        # 添加边到图
        print("🔗 Adding edges to graph...")
        for src, neighbors in tqdm(cooccur.items(), desc="Building graph edges"):
            for dst, weight in neighbors.items():
                if weight >= self.threshold:
                    self.graph.add_edge(src, dst, weight=weight)

        print(f"✅ Graph built with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
        return self.graph

    def generate_sequences(self, walk_length=20, num_walks=10):
        """生成随机游走序列"""
        print("\n🚶 Generating random walks...")
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
        self.visualizer = ProgressVisualizer()

    def train_graph_embeddings(self, behavior_df):
        """训练图增强的物品向量"""
        print("\n🔍 Training graph embeddings...")
        # 1. 构建关系图
        graph_builder = ItemGraphGenerator()
        item_graph = graph_builder.build_graph(behavior_df)

        # 2. 生成随机游走序列
        sequences = graph_builder.generate_sequences()

        # 3. 训练物品向量
        print("\n🎯 Training item vectors with Word2Vec...")
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
        print(f"✅ Item vectors trained for {len(self.item_vectors)} items")
        return self.item_vectors

    def build_joint_model(self):
        """构建联合模型架构"""
        print("\n🧩 Building joint model architecture...")
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
            inputs={**user_inputs, **item_inputs, 'item_graph_input': item_graph_input},
        outputs = output
        )

    def create_enhanced_dataset(self, data_path, batch_size=1024):
        """创建包含图特征的数据管道"""
        print(f"\n📊 Creating enhanced dataset from {data_path}...")
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

        print(f"✅ Dataset created with {len(data)} samples")
        return tf.data.Dataset.from_tensor_slices((
            {**user_features, ** item_features,**graph_features},
        labels
        )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_existing_model(model_dir):
    """加载预训练双塔模型"""
    print(f"\n⏳ Loading pre-trained models from {model_dir}...")
    user_tower = tf.keras.models.load_model(f'{model_dir}/user_tower')
    item_tower = tf.keras.models.load_model(f'{model_dir}/item_tower')
    print("✅ Models loaded successfully")
    return user_tower, item_tower

def main():
    # 初始化可视化工具
    visualizer = ProgressVisualizer()

    # 1. 加载现有模型
    user_tower, item_tower = load_existing_model('results/0427embed')

    # 2. 初始化增强系统
    enhanced_model = EnhancedDoubleTower(user_tower, item_tower)

    # 3. 训练图嵌入
    print("\n" + "=" * 50)
    print("Starting Graph Embedding Training Phase")
    print("=" * 50)
    behavior_data = pd.read_csv('data/cleaned_behavior.csv')
    enhanced_model.train_graph_embeddings(behavior_data)

    # 4. 构建联合模型
    print("\n" + "=" * 50)
    print("Starting Joint Model Construction")
    print("=" * 50)
    joint_model = enhanced_model.build_joint_model()
    joint_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    joint_model.summary()

    # 5. 准备数据
    print("\n" + "=" * 50)
    print("Preparing Training Data")
    print("=" * 50)
    train_ds = enhanced_model.create_enhanced_dataset('data/processed_data3.parquet')
    test_ds = enhanced_model.create_enhanced_dataset('data/processed_data_test3.parquet')

    # 6. 训练配置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f'checkpoints/double_tower_interest_graph/{timestamp}'
    os.makedirs(checkpoint_path, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f'{checkpoint_path}/enhanced_model_epoch{{epoch:02d}}.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/{timestamp}',
            histogram_freq=1
        )
    ]

    # 7. 微调训练
    print("\n" + "=" * 50)
    print("Starting Model Fine-Tuning")
    print("=" * 50)
    visualizer.start_timer()

    history = joint_model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    # 记录指标
    visualizer.record_metrics(history, 0)
    visualizer.plot_metrics()

    # 8. 保存最终模型
    print("\n" + "=" * 50)
    print("Saving Final Model")
    print("=" * 50)
    output_path = f'results/double_tower_interest_graph/{timestamp}'
    os.makedirs(output_path, exist_ok=True)
    joint_model.save(f'{output_path}/enhanced_final_model')

    # 保存训练曲线
    visualizer.plot_metrics()
    print(f"\n🎉 Training completed! Model saved to {output_path}")

if __name__ == "__main__":
    main()