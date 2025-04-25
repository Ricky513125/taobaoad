import tensorflow as tf
import pandas as pd
from model import build_user_tower, build_item_tower

# GPU配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def create_dataset(data_path, batch_size=1024, neg_ratio=4, is_train=True):
    """创建符合双塔结构的数据管道"""
    """创建数据管道，训练集需要负采样，测试集不需要"""
    data = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)

    # 分离用户特征和广告特征
    user_features = {
        'user_gender': data['final_gender_code'].values,
        'user_cms_segid': data['cms_segid'].values,
        'user_cms_group_id': data['cms_group_id'].values,
        'user_age_level': data['age_level'].values,
        'user_pvalue_level': data['pvalue_level'].values + 1,  # -1→0
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

    labels = data['clk'].values

    # 创建基础数据集
    dataset = tf.data.Dataset.from_tensor_slices((
        {**user_features, **item_features},  # 合并特征
    labels
    ))

    if is_train:
        # 训练集：Batch内负采样（与原始代码相同）
        def batch_neg_sampling(batch_features, batch_labels):
            pos_mask = batch_labels == 1
            pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32))

            pos_features = {
                k: tf.boolean_mask(v, pos_mask)
                for k, v in batch_features.items()
            }

            neg_indices = tf.random.shuffle(tf.range(tf.shape(batch_labels)[0]))[:pos_count * neg_ratio]
            neg_features = {
                k: tf.gather(v, neg_indices)
                for k, v in batch_features.items()
            }

            combined_features = {
                k: tf.concat([pos_features[k], neg_features[k]], axis=0)
                for k in batch_features.keys()
            }
            combined_labels = tf.concat([
                tf.ones_like(batch_labels[:pos_count], dtype=tf.float32),
                tf.zeros_like(batch_labels[:pos_count * neg_ratio], dtype=tf.float32)
            ], axis=0)

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
        # 测试集：直接返回原始数据（不需要负采样）
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Batch内负采样
    # def batch_neg_sampling(batch_features, batch_labels):
    #     pos_mask = batch_labels == 1
    #     pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32))
    #
    #     # 正样本
    #     pos_features = {
    #         k: tf.boolean_mask(v, pos_mask)
    #         for k, v in batch_features.items()
    #     }
    #
    #     # 负样本(从同batch随机选择)
    #     neg_indices = tf.random.shuffle(tf.range(tf.shape(batch_labels)[0]))[:pos_count * neg_ratio]
    #     neg_features = {
    #         k: tf.gather(v, neg_indices)
    #         for k, v in batch_features.items()
    #     }
    #
    #     # 合并正负样本
    #     combined_features = {
    #         k: tf.concat([pos_features[k], neg_features[k]], axis=0)
    #         for k in batch_features.keys()
    #     }
    #     combined_labels = tf.concat([
    #         tf.ones_like(batch_labels[:pos_count], dtype=tf.float32),
    #         tf.zeros_like(batch_labels[:pos_count * neg_ratio], dtype=tf.float32)
    #     ], axis=0)
    #
    #     # 打乱顺序
    #     indices = tf.random.shuffle(tf.range(tf.shape(combined_labels)[0]))
    #     return (
    #         {k: tf.gather(v, indices) for k, v in combined_features.items()},
    #         tf.gather(combined_labels, indices)
    #     )
    #
    # return (dataset
    #         .shuffle(100000)
    #         .batch(batch_size)
    #         .map(batch_neg_sampling, num_parallel_calls=tf.data.AUTOTUNE)
    #         .prefetch(tf.data.AUTOTUNE))


if __name__ == "__main__":
    # 1. 构建模型
    user_tower = build_user_tower()
    item_tower = build_item_tower()

    # 2. 组合双塔模型
    user_inputs = {k: v for k, v in user_tower.input.items()}
    item_inputs = {k: v for k, v in item_tower.input.items()}

    user_embed = user_tower(user_inputs)
    item_embed = item_tower(item_inputs)

    # 相似度计算
    dot_product = tf.keras.layers.Dot(axes=1, normalize=True)([user_embed, item_embed])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_product)

    model = tf.keras.Model(
        inputs={**user_inputs, **item_inputs},
    outputs = output
    )

    # 3. 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['AUC']
    )

    # 4. 训练配置
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/model_{epoch:02d}.h5',
        save_weights_only=False,  # 保存完整模型
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    # 5. 训练
    train_ds = create_dataset('data/processed_data.parquet',is_train=True)
    test_ds = create_dataset('data/raw_sample_test.parquet', is_train=False)  # 测试集
    history = model.fit(
        train_ds,
        validation_data=test_ds,  # 加入测试集作为验证集
        epochs=100,
        callbacks=[checkpoint_callback],
        verbose=1
    )
    # 7. 最终评估测试集
    test_loss, test_auc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest AUC: {test_auc:.4f}")

    # 6. 保存最终模型
    model.save('final_model')
    user_tower.save('user_tower')
    item_tower.save('item_tower')