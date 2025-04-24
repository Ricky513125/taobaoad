




import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 加载数据
ad_feature = pd.read_csv('ad_feature.csv')
user_profile = pd.read_csv('user_profile.csv')

# 合并正样本数据（假设raw_sample已处理）
# raw_sample包含user_id, adgroup_id, label(是否点击)
raw_sample = pd.read_csv('raw_sample.csv')
data = pd.merge(raw_sample, user_profile, left_on='user_id', right_on='userid')
data = pd.merge(data, ad_feature, on='adgroup_id')

# 处理缺失值
data['pvalue_level'].fillna(-1, inplace=True)
data['new_user_class_level'].fillna(0, inplace=True)

# 处理异常值
data['price'] = np.where(data['price'] <= 0, data['price'].median(), data['price'])

# 定义特征分桶
def bucketize(series, num_buckets=10):
    return pd.qcut(series, num_buckets, labels=False, duplicates='drop')

data['price_bucket'] = bucketize(data['price'], num_buckets=5)

# 生成负样本 (1:4 正负样本比例)
neg_samples = data[data['label'] == 0].sample(frac=0.8)
pos_samples = data[data['label'] == 1]
dataset = pd.concat([pos_samples, neg_samples]).sample(frac=1)


####
# 开始构建双塔模型
def build_tower(input_dims, embedding_dims, hidden_units, name):
    """构建塔结构"""
    inputs = {}
    embeddings = []

    # 分类特征嵌入
    for feat, dim in input_dims['categorical'].items():
        input_layer = tf.keras.layers.Input(shape=(1,), name=f"{name}_{feat}")
        inputs[feat] = input_layer
        embeddings.append(
            tf.keras.layers.Embedding(
                input_dim=dim,
                output_dim=embedding_dims,
                name=f"{name}_embed_{feat}"
            )(input_layer)
        )

    # 数值特征
    for feat in input_dims['numerical']:
        input_layer = tf.keras.layers.Input(shape=(1,), name=f"{name}_{feat}")
        inputs[feat] = input_layer
        embeddings.append(tf.expand_dims(input_layer, axis=-1))

    # 拼接所有特征
    concat = tf.keras.layers.Concatenate(axis=-1)(embeddings)

    # DNN部分
    dnn = tf.keras.layers.BatchNormalization()(concat)
    for units in hidden_units:
        dnn = tf.keras.layers.Dense(units, activation='relu')(dnn)
        dnn = tf.keras.layers.BatchNormalization()(dnn)
    output = tf.keras.layers.Dense(64, activation=None)(dnn)  # 输出向量维度64

    return tf.keras.Model(inputs=inputs, outputs=output, name=name)


# 用户塔特征配置
user_tower = build_tower(
    input_dims={
        'categorical': {
            'gender': 3,  # 0:缺失,1:男,2:女
            'age_level': 9,
            'pvalue_level': 4,  # -1,1,2,3
            'shopping_level': 4,
            'city_level': 7
        },
        'numerical': ['occupation']
    },
    embedding_dims=8,
    hidden_units=[128, 64],
    name='user_tower'
)

# 广告塔特征配置
item_tower = build_tower(
    input_dims={
        'categorical': {
            'cate_id': 1000,
            'brand': 5000,
            'price_bucket': 6
        },
        'numerical': []
    },
    embedding_dims=16,
    hidden_units=[256, 128],
    name='item_tower'
)

# 组合双塔
user_inputs = {k: v for k, v in user_tower.input.items()}
item_inputs = {k: v for k, v in item_tower.input.items()}

user_embedding = user_tower(user_inputs)
item_embedding = item_tower(item_inputs)

# 计算相似度
dot_product = tf.keras.layers.Dot(axes=1, normalize=True)([user_embedding, item_embedding])
output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_product)

model = tf.keras.Model(inputs={**user_inputs,** item_inputs}, outputs = output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])


######
## 数据输入管道
# 定义特征列转换
def df_to_dataset(data, batch_size=1024):
    return tf.data.Dataset.from_tensor_slices({
        # 用户特征
        'gender': data['final_gender_code'].values,
        'age_level': data['age_level'].values,
        'pvalue_level': data['pvalue_level'].values + 1,  # -1变为0
        'shopping_level': data['shopping_level'].values,
        'city_level': data['new_user_class_level'].values,
        'occupation': data['occupation'].values,
        # 广告特征
        'cate_id': data['cate_id'].values,
        'brand': data['brand'].values,
        'price_bucket': data['price_bucket'].values
    }).shuffle(100000).batch(batch_size)

# 划分训练/验证集
train_df, val_df = train_test_split(dataset, test_size=0.2)
train_ds = df_to_dataset(train_df)
val_ds = df_to_dataset(val_df)


#####
## 模型评估
# 训练配置
early_stop = tf.keras.callbacks.EarlyStopping(patience=3)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop, reduce_lr]
)

# 保存用户塔用于线上服务
user_tower.save('user_tower.h5')



# 加载模型
user_tower = tf.keras.models.load_model('user_tower.h5')

# 生成用户向量
def get_user_vector(user_features):
    return user_tower.predict(user_features)

# 物料向量预计算（离线）
item_vectors = item_tower.predict(item_features)
np.save('item_vectors.npy', item_vectors)

# 线上召回（Faiss实现）
import faiss

index = faiss.IndexFlatIP(64)  # 内积相似度
index.add(item_vectors)

def recall(user_vec, top_k=100):
    distances, indices = index.search(user_vec, top_k)
    return indices[0]