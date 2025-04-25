import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Concatenate, Dense
from tensorflow.keras.models import Model


class SqueezeLayer(Layer):
    """自定义Keras层实现squeeze操作"""

    def __init__(self, axis=None, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)


def build_user_tower(embedding_dim=8, hidden_units=[256, 128], name='user_tower'):
    """构建用户塔（User Tower 模型）"""
    input_dims = {
        'categorical': {
            'user_cms_segid': 100,
            'user_cms_group_id': 50,
            'user_gender': 3,
            'user_age_level': 8,
            'user_pvalue_level': 4,
            'user_shopping_level': 4,
            'user_city_level': 5
        },
        'numerical': ['user_occupation']  # occupation 是数值型，0或1
    }

    # 1. 构建输入层
    inputs = {}
    for feat, dim in input_dims['categorical'].items():
        inputs[feat] = Input(shape=(1,), name=feat, dtype=tf.int32)
    for feat in input_dims['numerical']:
        inputs[feat] = Input(shape=(1,), name=feat, dtype=tf.float32)

    # 2. 嵌入层处理 categorical 特征
    embeddings = []
    for feat, dim in input_dims['categorical'].items():
        embed = Embedding(input_dim=dim, output_dim=embedding_dim, name=f'embedding_{feat}')(inputs[feat])
        embeddings.append(SqueezeLayer(axis=1)(embed))  # 使用自定义层替代tf.squeeze

    # 3. 添加 numerical 特征
    for feat in input_dims['numerical']:
        embeddings.append(inputs[feat])  # 数值型直接拼接

    # 4. 拼接 + DNN 层
    concat = Concatenate()(embeddings)
    x = concat
    for units in hidden_units:
        x = Dense(units, activation='relu')(x)

    return Model(inputs=inputs, outputs=x, name=name)


def build_item_tower(embedding_dim=16, hidden_units=[256, 128], name='item_tower'):
    """构建广告塔（Item Tower）"""
    # 1. 构建输入层（统一数值类型）
    inputs = {
        'item_adgroup_id': Input(shape=(1,), name='adgroup_id', dtype=tf.int32),
        'item_cate_id': Input(shape=(1,), name='cate_id', dtype=tf.int32),
        'item_campaign_id': Input(shape=(1,), name='campaign_id', dtype=tf.int32),
        'item_customer': Input(shape=(1,), name='customer', dtype=tf.int32),
        'item_brand': Input(shape=(1,), name='brand', dtype=tf.float32),  # 改为int32类型
        'item_price': Input(shape=(1,), name='price', dtype=tf.float32)
    }

    # 2. 嵌入层处理（使用自定义SqueezeLayer）
    embeddings = [
        SqueezeLayer(axis=1)(Embedding(1000000, embedding_dim, name='embed_adgroup')(inputs['item_adgroup_id'])),
        SqueezeLayer(axis=1)(Embedding(1000, embedding_dim, name='embed_cate')(inputs['item_cate_id'])),
        SqueezeLayer(axis=1)(Embedding(100000, embedding_dim, name='embed_campaign')(inputs['item_campaign_id'])),
        SqueezeLayer(axis=1)(Embedding(500000, embedding_dim, name='embed_customer')(inputs['item_customer'])),
        SqueezeLayer(axis=1)(Embedding(10000, embedding_dim, name='embed_brand')(inputs['item_brand'])),  # brand改为嵌入
    ]

    # 3. 数值特征处理
    embeddings.append(inputs['item_price'])  # 价格保持float32

    # 4. DNN 部分
    concat = Concatenate()(embeddings)
    x = concat
    for units in hidden_units:
        x = Dense(units, activation='relu')(x)

    return Model(inputs=inputs, outputs=x, name=name)

def load_models(model_dir):
    """加载预训练模型"""
    user_tower = UserTower()
    user_tower.load_weights(f"{model_dir}/user_tower")
    item_tower = ItemTower()
    item_tower.load_weights(f"{model_dir}/item_tower")
    return user_tower, item_tower