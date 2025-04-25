import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Concatenate, Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import os


class SqueezeLayer(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        config = super(SqueezeLayer, self).get_config()
        config.update({'axis': self.axis})
        return config


class SequenceAugmentedEmbedding(Layer):
    """融合行为序列的增强Embedding层（支持GPU加速）"""

    def __init__(self, input_dim, output_dim, sequence_emb_dim=8, l2_reg=0.01, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.original_embedding = Embedding(
            input_dim, output_dim,
            embeddings_regularizer=l2(l2_reg)
        )
        self.sequence_projection = Dense(
            output_dim, activation='tanh',
            kernel_regularizer=l2(l2_reg)
        )
        self.alpha = self.add_weight(
            name='fusion_alpha',
            shape=(),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            item_id, seq_emb = inputs
            original_emb = self.original_embedding(item_id)
            seq_emb = self.sequence_projection(seq_emb)
            alpha = tf.sigmoid(self.alpha)
            return alpha * original_emb + (1 - alpha) * seq_emb

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        })
        return config


class MultiHeadSequenceAttention(Layer):
    """GPU优化的多头注意力"""

    def __init__(self, num_heads=4, head_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def build(self, input_shape):
        self.query = Dense(self.num_heads * self.head_dim)
        self.key = Dense(self.num_heads * self.head_dim)
        self.value = Dense(self.num_heads * self.head_dim)
        self.combine = Dense(input_shape[-1])

    def call(self, inputs):
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            batch_size = tf.shape(inputs)[0]

            q = self.query(inputs)
            k = self.key(inputs)
            v = self.value(inputs)

            q = tf.reshape(q, [batch_size, -1, self.num_heads, self.head_dim])
            k = tf.reshape(k, [batch_size, -1, self.num_heads, self.head_dim])
            v = tf.reshape(v, [batch_size, -1, self.num_heads, self.head_dim])

            scores = tf.einsum('bqhd,bkhd->bhqk', q, k) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
            attn = tf.nn.softmax(scores, axis=-1)

            out = tf.einsum('bhql,blhd->bqhd', attn, v)
            out = tf.reshape(out, [batch_size, -1, self.num_heads * self.head_dim])
            return self.combine(out)


def build_user_tower(embedding_dim=64, seq_emb_dim=32, hidden_units=[256, 128], name='user_tower'):
    """GPU优化的用户塔"""
    inputs = {
        'user_cms_segid': Input(shape=(1,), dtype=tf.int32),
        'user_gender': Input(shape=(1,), dtype=tf.int32),
        'user_age_level': Input(shape=(1,), dtype=tf.int32),
        'hist_cate_seq': Input(shape=(None,), dtype=tf.int32),
        'hist_brand_seq': Input(shape=(None,), dtype=tf.int32)
    }

    # 序列处理（GPU加速）
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        cate_emb = Embedding(1000, seq_emb_dim, mask_zero=True)(inputs['hist_cate_seq'])
        brand_emb = Embedding(10000, seq_emb_dim, mask_zero=True)(inputs['hist_brand_seq'])
        seq_emb = Concatenate()([cate_emb, brand_emb])
        seq_emb = MultiHeadSequenceAttention()(seq_emb)
        seq_emb = LSTM(seq_emb_dim)(seq_emb)
        seq_emb = Dense(embedding_dim, activation='tanh')(seq_emb)

    # 特征处理
    embeddings = []
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # 重要特征使用增强Embedding
        emb = SequenceAugmentedEmbedding(100, embedding_dim)(
            [inputs['user_cms_segid'], seq_emb])
        embeddings.append(SqueezeLayer(axis=1)(emb))

        # 其他特征
        embeddings.append(SqueezeLayer(axis=1)(
            Embedding(3, embedding_dim)(inputs['user_gender'])))

        # 数值特征
        age_emb = Dense(embedding_dim)(tf.cast(inputs['user_age_level'], tf.float32))
        embeddings.append(SqueezeLayer(axis=1)(age_emb))

    # DNN部分
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        x = Concatenate()(embeddings)
        for units in hidden_units:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)

    return Model(inputs=inputs, outputs=x, name=name)


def build_item_tower(embedding_dim=64, seq_emb_dim=64, hidden_units=[256, 128], name='item_tower'):
    """GPU优化的物品塔"""
    inputs = {
        'item_cate_id': Input(shape=(1,), dtype=tf.int32),
        'item_brand': Input(shape=(1,), dtype=tf.int32),
        'graph_seq': Input(shape=(None,), dtype=tf.int32)
    }

    # 序列处理
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        seq_emb = Embedding(1000000, seq_emb_dim, mask_zero=True)(inputs['graph_seq'])
        seq_emb = MultiHeadSequenceAttention()(seq_emb)
        seq_emb = LSTM(seq_emb_dim)(seq_emb)
        seq_emb = Dense(embedding_dim, activation='tanh')(seq_emb)

    # 特征处理
    embeddings = []
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # 重要特征使用增强Embedding
        emb = SequenceAugmentedEmbedding(1000, embedding_dim)(
            [inputs['item_cate_id'], seq_emb])
        embeddings.append(SqueezeLayer(axis=1)(emb))

        emb = SequenceAugmentedEmbedding(10000, embedding_dim)(
            [inputs['item_brand'], seq_emb])
        embeddings.append(SqueezeLayer(axis=1)(emb))

    # DNN部分
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        x = Concatenate()(embeddings)
        for units in hidden_units:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)

    return Model(inputs=inputs, outputs=x, name=name)


def build_two_tower_model(user_tower, item_tower):
    """构建完整双塔模型（支持GPU）"""
    # 合并输入
    inputs = {**user_tower.input,**item_tower.input}

    # 计算相似度
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        user_emb = user_tower({k: v for k, v in inputs.items() if k in user_tower.input})
        item_emb = item_tower({k: v for k, v in inputs.items() if k in item_tower.input})
        logits = tf.reduce_sum(user_emb * item_emb, axis=1, keepdims=True)
        output = tf.sigmoid(logits)

    return Model(inputs=inputs, outputs=output, name='two_tower_model')


def get_callbacks(log_dir, checkpoint_path):
    """获取训练回调（支持GPU）"""
    return [
        ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='batch'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]