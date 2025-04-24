import tensorflow as tf
from tensorflow.keras.layers import *


def build_user_tower(embedding_dim=8, hidden_units=[256, 128], name='user_tower'):
    """构建用户塔（User Tower 模型）"""

    input_dims = {
        'categorical': {
            'cms_segid': 100,
            'cms_group': 50,
            'gender': 3,
            'age_level': 8,
            'pvalue_level': 4,
            'shopping_level': 4,
            'city_level': 5
        },
        'numerical': ['occupation']  # occupation 是数值型，0或1
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
        embeddings.append(tf.squeeze(embed, axis=1))  # squeeze 掉 shape 中的 1

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

    # 1. 构建输入层
    inputs = {
        'adgroup_id': Input(shape=(1,), name='adgroup_id', dtype=tf.int32),
        'cate_id': Input(shape=(1,), name='cate_id', dtype=tf.int32),
        'campaign_id': Input(shape=(1,), name='campaign_id', dtype=tf.int32),
        'customer': Input(shape=(1,), name='customer', dtype=tf.int32),
        'brand': Input(shape=(1,), name='brand', dtype=tf.float32),
        'price': Input(shape=(1,), name='price', dtype=tf.float32)
    }

    # 2. 嵌入层处理 int64 类别特征（你可以根据数据大小调整input_dim）
    embeddings = [
        tf.squeeze(Embedding(input_dim=1000000, output_dim=embedding_dim, name='embed_adgroup')(inputs['adgroup_id']), axis=1),
        tf.squeeze(Embedding(input_dim=1000, output_dim=embedding_dim, name='embed_cate')(inputs['cate_id']), axis=1),
        tf.squeeze(Embedding(input_dim=100000, output_dim=embedding_dim, name='embed_campaign')(inputs['campaign_id']), axis=1),
        tf.squeeze(Embedding(input_dim=500000, output_dim=embedding_dim, name='embed_customer')(inputs['customer']), axis=1),
    ]

    # 3. 数值特征直接拼接（可以归一化后使用）
    embeddings.append(inputs['brand'])
    embeddings.append(inputs['price'])

    # 4. DNN 部分
    concat = Concatenate()(embeddings)
    x = concat
    for units in hidden_units:
        x = Dense(units, activation='relu')(x)

    return Model(inputs=inputs, outputs=x, name=name)