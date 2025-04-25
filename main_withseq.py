import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from models.model_rw import build_user_tower, build_item_tower, build_two_tower_model, get_callbacks
from models.graph_argument import build_item_graph, generate_sequences
import argparse

# GPU配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)


def create_dataset(data_path, batch_size=1024, neg_ratio=4, is_train=True, max_seq_len=20):
    """创建包含行为序列的数据管道"""
    # 读取数据
    data = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)

    # 处理行为序列
    def process_hist_sequence(seq_str):
        if pd.isna(seq_str):
            return np.zeros(max_seq_len, dtype=np.int32)
        seq = [int(x.split(',')[0]) for x in seq_str.split('|')[:max_seq_len]]
        return np.array(seq + [0] * (max_seq_len - len(seq)), dtype=np.int32)

    # 用户特征（包含行为序列）
    user_features = {
        'user_gender': data['final_gender_code'].values.astype('int32'),
        'user_cms_segid': data['cms_segid'].values.astype('int32'),
        'user_cms_group_id': data['cms_group_id'].values.astype('int32'),
        'user_age_level': data['age_level'].values.astype('int32'),
        'user_pvalue_level': (data['pvalue_level'].values).astype('int32'),  # -1→0
        'user_shopping_level': data['shopping_level'].values.astype('int32'),
        'user_city_level': data['new_user_class_level'].values.astype('int32'),
        'user_occupation': data['occupation'].values.astype('float32'),
        'hist_cate_seq': np.stack(data['hist_cate_seq'].apply(process_hist_sequence)),
        'hist_brand_seq': np.stack(data['hist_brand_seq'].apply(process_hist_sequence))
    }

    # 物品特征（包含图序列）
    item_features = {
        'item_adgroup_id': data['adgroup_id'].values.astype('int32'),
        'item_cate_id': data['cate_id'].values.astype('int32'),
        'item_campaign_id': data['campaign_id'].values.astype('int32'),
        'item_customer': data['customer'].values.astype('int32'),
        'item_brand': data['brand'].values.astype('int32'),
        'item_price': data['price'].values.astype('float32'),
        'graph_seq': np.stack(data['graph_seq'].apply(process_hist_sequence))
    }

    labels = data['clk'].values.astype('float32')

    # 创建基础数据集
    dataset = tf.data.Dataset.from_tensor_slices((
        {**user_features, **item_features},
    labels
    ))

    if is_train:
        # 训练集：Batch内负采样
        def batch_neg_sampling(batch_features, batch_labels):
            pos_mask = batch_labels == 1
            pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32))

            # 正样本
            pos_features = {
                k: tf.boolean_mask(v, pos_mask)
                for k, v in batch_features.items()
            }

            # 负样本(从同batch随机选择)
            neg_indices = tf.random.shuffle(tf.range(tf.shape(batch_labels)[0]))[:pos_count * neg_ratio]
            neg_features = {
                k: tf.gather(v, neg_indices)
                for k, v in batch_features.items()
            }

            # 合并正负样本
            combined_features = {
                k: tf.concat([pos_features[k], neg_features[k]], axis=0)
                for k in batch_features.keys()
            }
            combined_labels = tf.concat([
                tf.ones(pos_count, dtype=tf.float32),
                tf.zeros(pos_count * neg_ratio, dtype=tf.float32)
            ], axis=0)

            # 打乱顺序
            indices = tf.random.shuffle(tf.range(tf.shape(combined_labels)[0]))
            return (
                {k: tf.gather(v, indices) for k, v in combined_features.items()},
                tf.gather(combined_labels, indices)
            )

        return (dataset
                .shuffle(100000)
                .batch(batch_size)
                .map(batch_neg_sampling, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    else:
        # 测试集：直接返回原始数据
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_model(args):
    """训练主函数"""
    # 1. 准备数据
    print("Preparing data...")
    train_dataset = create_dataset(args.train_data, is_train=True)
    val_dataset = create_dataset(args.val_data, is_train=False) if args.val_data else None

    # 2. 构建图数据（如果不存在）
    if not os.path.exists(args.graph_path):
        print("Building item graph...")
        build_item_graph(args.behavior_data, args.graph_path)

    if not os.path.exists(args.sequence_path):
        print("Generating sequences...")
        with open(args.graph_path, 'rb') as f:
            graph = pickle.load(f)
        generate_sequences(graph, args.sequence_path)

    # 3. 构建模型
    print("Building models...")
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()

    with strategy.scope():
        user_tower = build_user_tower(
            embedding_dim=args.embedding_dim,
            seq_emb_dim=args.seq_emb_dim
        )
        item_tower = build_item_tower(
            embedding_dim=args.embedding_dim,
            seq_emb_dim=args.seq_emb_dim
        )
        model = build_two_tower_model(user_tower, item_tower)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

    # 4. 准备回调
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"model_{timestamp}")

    callbacks = get_callbacks(log_dir, checkpoint_path)

    # 5. 训练模型
    print("Start training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # 6. 保存模型
    print("Saving models...")
    user_tower.save(os.path.join(args.model_dir, "user_tower"))
    item_tower.save(os.path.join(args.model_dir, "item_tower"))
    model.save(os.path.join(args.model_dir, "full_model"))

    # 7. 评估测试集
    if args.test_data:
        print("Evaluating on test set...")
        test_dataset = create_dataset(args.test_data, is_train=False)
        test_results = model.evaluate(test_dataset, verbose=1)
        print("\nTest Results:")
        for name, value in zip(model.metrics_names, test_results):
            print(f"{name}: {value:.4f}")

    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Two-Tower Model with Sequence Enhancement')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_data', type=str, default=None, help='Path to validation data')
    parser.add_argument('--test_data', type=str, default=None, help='Path to test data')
    parser.add_argument('--behavior_data', type=str, required=True, help='Path to behavior data for graph building')
    parser.add_argument('--graph_path', type=str, default='./graph.pkl', help='Path to save/load graph')
    parser.add_argument('--sequence_path', type=str, default='./sequences.txt', help='Path to save/load sequences')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension for features')
    parser.add_argument('--seq_emb_dim', type=int, default=32, help='Embedding dimension for sequences')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--neg_ratio', type=int, default=4, help='Negative sample ratio')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for TensorBoard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory for model checkpoints')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory for saved models')

    args = parser.parse_args()

    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    train_model(args)