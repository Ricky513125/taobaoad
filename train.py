import tensorflow as tf
import numpy as np
from model import build_user_tower, build_item_tower
from sklearn.model_selection import train_test_split


class SqueezeLayer(Layer):
    """自定义Keras层实现squeeze操作"""

    def __init__(self, axis=None, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

def create_dataset(data_path):
    """创建动态负采样数据管道"""
    data = pd.read_parquet(data_path)

    def in_batch_negative_sampling(batch, neg_ratio=4):
        """Batch内负采样"""
        pos_mask = batch['clk'] == 1
        pos_data = {k: tf.boolean_mask(v, pos_mask) for k, v in batch.items()}

        # 生成负样本索引
        neg_indices = tf.random.shuffle(tf.range(tf.shape(batch['clk'])[0]))[:tf.shape(pos_data['clk'])[0] * neg_ratio]
        neg_data = {k: tf.gather(v, neg_indices) for k, v in batch.items()}

        # 合并正负样本
        combined = {k: tf.concat([pos_data[k], neg_data[k]], axis=0) for k in batch.keys()}
        shuffled_indices = tf.random.shuffle(tf.range(tf.shape(combined['clk'])[0]))
        return {k: tf.gather(v, shuffled_indices) for k, v in combined.items()}

    # 定义特征转换
    features = {
        'gender': tf.cast(data['final_gender_code'], tf.int32),
        'cms_segid': tf.cast(data['cms_segid'], tf.int32),
        'cms_group': tf.cast(data['cms_group'], tf.int32),
        'age_level': tf.cast(data['age_level'], tf.int32),
        'pvalue_level': tf.cast(data['pvalue_level'], tf.int32),  # -1→0
        'shopping_level': tf.cast(data['shopping_level'], tf.int32),
        'city_level': tf.cast(data['new_user_class_level'], tf.int32),
        'occupation': tf.cast(data['occupation'], tf.float32),
        'cate_id': tf.cast(data['cate_id'], tf.int32),
        'brand': tf.cast(data['brand'], tf.float32),
        # 'price_bucket': tf.cast(data['price_bucket'], tf.int32),
        'campaign_id': tf.cast(data['campaign_id'], tf.float32),
        'customer': tf.cast(data['customer'], tf.int32),
        'price': tf.cast(data['price'], tf.float32),
        'clk': tf.cast(data['clk'], tf.int32)
    }

    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.shuffle(100000).batch(1024).map(in_batch_negative_sampling)
    return ds


if __name__ == "__main__":
    # 构建模型
    user_tower = build_user_tower()
    item_tower = build_item_tower()

    # 计算相似度
    dot_product = tf.keras.layers.Dot(axes=1, normalize=True)([user_tower.output, item_tower.output])
    output = Dense(1, activation='sigmoid')(dot_product)
    model = tf.keras.Model(inputs=[user_tower.input, item_tower.input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    # ========== 2. 配置检查点回调 ==========
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = f"{checkpoint_dir}/dual_tower_epoch"

    # 每2个epoch保存一次，只保留最新的3个检查点
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix + "{epoch:02d}",
        save_weights_only=True,  # 仅保存权重（节省空间）
        save_freq='epoch',  # 按epoch保存
        period=2,  # 每2个epoch保存一次
        max_to_keep=3  # 最多保留3个检查点
    )

    # ========== 3. 训练模型 ==========
    train_ds = create_dataset('processed_data.parquet')
    history = model.fit(
        train_ds,
        epochs=10,
        callbacks=[checkpoint_callback]  # 添加检查点回调
    )

    # ========== 4. 最终模型保存 ==========
    # 保存完整模型（可选）
    model.save('final_model.h5')  # HDF5格式

    # 单独保存双塔（用于线上服务）
    user_tower.save('user_tower.h5')
    item_tower.save('item_tower.h5')