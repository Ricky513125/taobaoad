import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Dot
from tensorflow.keras import Model
# rank/train_rank.py
from utils.feature_utils import build_rank_features
def build_deepfm(feature_config):
    # 数值型特征输入
    numeric_inputs = {
        name: Input(shape=(1,), dtype='float32', name=name)
        for name in feature_config['numeric_features']
    }

    # 类别型特征输入
    categorical_inputs = {
        name: Input(shape=(1,), dtype='int32', name=name)
        for name in feature_config['categorical_features']
    }

    # 1. FM部分（二阶交叉）
    categorical_embeddings = []
    for name, vocab_size in feature_config['categorical_vocab'].items():
        emb = Embedding(vocab_size, 8, name=f'emb_{name}')(categorical_inputs[name])
        categorical_embeddings.append(emb)

    # 特征交叉
    cross_terms = []
    for i in range(len(categorical_embeddings)):
        for j in range(i+1, len(categorical_embeddings)):
            cross = Dot(axes=1, normalize=False)([categorical_embeddings[i], categorical_embeddings[j]])
            cross_terms.append(cross)

    # 2. DNN部分
    dnn_input = Concatenate()([
        *[tf.reshape(emb, (-1, 8)) for emb in categorical_embeddings],
        *[numeric_inputs[name] for name in feature_config['numeric_features']]
    ])
    dnn_output = Dense(128, activation='relu')(dnn_input)
    dnn_output = Dense(64, activation='relu')(dnn_output)

    # 3. 合并输出
    combined = Concatenate()([*cross_terms, dnn_output])
    output = Dense(1, activation='sigmoid', name='prediction')(combined)

    return Model(
        inputs=[*numeric_inputs.values(), *categorical_inputs.values()],
        outputs=output
    )



# 加载召回结果和原始特征
recall_results = pd.read_parquet('recall/output/recall_results.parquet')
user_features = load_user_features()  # 自定义加载函数
item_features = load_item_features()

# 生成精排训练数据
rank_data = build_rank_features(recall_results, user_features, item_features)
train_df, val_df = train_test_split(rank_data, test_size=0.2)

# 定义特征配置
feature_config = {
    'numeric_features': ['item_price', 'cosine_sim', 'price_sensitivity'],
    'categorical_features': ['user_gender', 'item_cate_id'],
    'categorical_vocab': {
        'user_gender': 3,
        'item_cate_id': 100
    }
}

# 训练模型
model = build_deepfm(feature_config)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

train_ds = df_to_dataset(train_df, feature_config)  # 自定义数据转换函数
val_ds = df_to_dataset(val_df, feature_config)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[EarlyStopping(patience=3)]
)