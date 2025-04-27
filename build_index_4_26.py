"""
4.27 success
"""

import os
import faiss
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import time
import gc
from multiprocessing import cpu_count
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model

# GPU配置初始化
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)


class SqueezeLayer(tf.keras.layers.Layer):
    """保持与您的模型完全一致"""

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)


def load_item_model(model_path):
    """完全修复变量引用问题的模型加载函数"""
    try:
        # 显式指定GPU设备
        with tf.device('/GPU:0'):
            # 检查是否是SavedModel格式目录
            if os.path.isdir(model_path):
                print("加载SavedModel格式的item_tower...")

                # 使用tf.saved_model.load加载模型
                model = tf.saved_model.load(model_path)

                # 获取serving签名并确保持久化
                if 'serving_default' not in model.signatures:
                    raise ValueError("SavedModel中没有找到serving_default签名")

                serve = model.signatures['serving_default']
                output_name = list(serve.structured_outputs.keys())[0]
                input_names = list(serve.structured_input_signature[1].keys())

                # 创建持久化包装类
                class ModelWrapper:
                    def __init__(self, model, serve, output_name):
                        # 关键：将模型和签名存储在实例属性中
                        self._model = model  # 保持模型引用
                        self._serve = serve  # 保持签名引用
                        self._output_name = output_name

                    @tf.function
                    def __call__(self, inputs):
                        # 构建输入字典（根据实际输入调整）
                        input_dict = {
                            'adgroup_id': tf.cast(inputs['item_adgroup_id'], tf.int32),
                            'cate_id': tf.cast(inputs['item_cate_id'], tf.int32),
                            'campaign_id': tf.cast(inputs['item_campaign_id'], tf.int32),
                            'customer': tf.cast(inputs['item_customer'], tf.int32),
                            'brand': tf.cast(inputs['item_brand'], tf.float32),
                            'price': inputs['item_price']
                        }
                        # 确保使用持久化的serve引用
                        return self._serve(**input_dict)[self._output_name]

                return ModelWrapper(model, serve, output_name)
            else:
                raise ValueError("模型路径不是有效的SavedModel目录")
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")


def create_dataset(item_data, batch_size=1024):
    """创建TF Dataset管道（GPU优化）"""

    # 生成符合模型输入结构的字典
    def gen():
        for _, row in item_data.iterrows():
            yield {
                'item_adgroup_id': np.array([row['adgroup_id']], dtype=np.int32),
                'item_cate_id': np.array([row['cate_id']], dtype=np.int32),
                'item_campaign_id': np.array([row['campaign_id']], dtype=np.int32),
                'item_customer': np.array([row['customer']], dtype=np.int32),
                'item_brand': np.array([row['brand']], dtype=np.float32),
                'item_price': np.array([row['price']], dtype=np.float32)
            }

    output_signature = {
        'item_adgroup_id': tf.TensorSpec(shape=(1,), dtype=tf.int32),
        'item_cate_id': tf.TensorSpec(shape=(1,), dtype=tf.int32),
        'item_campaign_id': tf.TensorSpec(shape=(1,), dtype=tf.int32),
        'item_customer': tf.TensorSpec(shape=(1,), dtype=tf.int32),
        'item_brand': tf.TensorSpec(shape=(1,), dtype=tf.float32),
        'item_price': tf.TensorSpec(shape=(1,), dtype=tf.float32)
    }

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def generate_vectors_gpu(model, item_data, batch_size=8192):
    """GPU加速的向量生成"""
    dataset = create_dataset(item_data, batch_size)

    # 预分配GPU内存
    num_items = len(item_data)
    sample_output = model(next(iter(dataset.take(1))))
    vector_dim = sample_output.shape[-1]

    # 使用锁页内存(pinned memory)加速CPU-GPU传输
    vectors = np.zeros((num_items, vector_dim), dtype=np.float32)

    # 多进程数据加载+GPU预测
    @tf.function(experimental_relax_shapes=True)
    def predict_fn(batch):
        return model(batch)

    start_idx = 0
    for batch in tqdm(dataset, total=(num_items // batch_size) + 1, desc="GPU预测"):
        batch_vectors = predict_fn(batch).numpy()
        end_idx = start_idx + len(batch_vectors)
        vectors[start_idx:end_idx] = batch_vectors
        start_idx = end_idx

    return vectors


def build_faiss_index(vectors, ids, output_dir="recall"):
    """纯CPU版本的索引构建"""
    os.makedirs(output_dir, exist_ok=True)

    # 归一化向量（重要步骤）
    faiss.normalize_L2(vectors)

    # 根据数据规模自动选择索引类型
    dim = vectors.shape[1]
    if len(vectors) < 1_000_000:
        # 小数据集使用精确搜索
        index = faiss.IndexFlatIP(dim)  # 内积距离
        print("使用精确搜索(IndexFlatIP)")
    else:
        # 大数据集使用IVF近似搜索
        nlist = min(10000, len(vectors) // 10)  # 聚类中心数
        quantizer = faiss.IndexFlatIP(dim)  # 量化器
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"使用IVF近似搜索，nlist={nlist}")

        # 需要先训练索引
        print("训练索引中...")
        index.train(vectors)

    # 添加向量到索引
    print("添加向量到索引...")
    index.add(vectors)
    print(f"索引构建完成，包含 {index.ntotal} 个向量")

    # 版本化保存
    timestamp = int(time.time())
    index_path = os.path.join(output_dir, f"item_index_{timestamp}.faiss")
    ids_path = os.path.join(output_dir, f"item_ids_{timestamp}.npy")

    # 保存索引和ID映射
    faiss.write_index(index, index_path)
    np.save(ids_path, ids)
    print(f"索引已保存到: {index_path}")
    print(f"ID映射已保存到: {ids_path}")

    # 创建最新版本的软链接
    latest_index_path = os.path.join(output_dir, "item_index_latest.faiss")
    if os.path.exists(latest_index_path):
        os.remove(latest_index_path)
    os.symlink(os.path.basename(index_path), latest_index_path)
    print(f"创建软链接: {latest_index_path} -> {os.path.basename(index_path)}")

    return index_path, ids_path


if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "item_tower"  # 这是包含saved_model.pb的目录
    ITEM_DATA_PATH = "data/ad_feature.csv"

    try:
        # 1. 加载数据
        print("加载物料数据...")
        item_data = pd.read_csv(ITEM_DATA_PATH)
        print(f"数据量: {len(item_data):,} 条")

        # 2. 加载模型
        print("\n加载item_tower模型...")
        item_tower = load_item_model(MODEL_PATH)

        # 3. GPU向量生成
        print("\n启动GPU加速生成向量...")
        start_time = time.time()
        item_vectors = generate_vectors_gpu(item_tower, item_data)
        print(f"生成完成! 耗时: {time.time() - start_time:.2f}秒")
        print(f"向量维度: {item_vectors.shape}")

        # 4. 构建索引
        print("\n构建FAISS索引...")
        index_path, ids_path = build_faiss_index(item_vectors, item_data['adgroup_id'].values)

        print(f"\n构建完成！")
        print(f"索引文件: {os.path.abspath(index_path)}")
        print(f"ID映射文件: {os.path.abspath(ids_path)}")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("调试建议:")
        print("1. 运行 nvidia-smi 检查GPU状态")
        print("2. 确认 tensorflow-gpu 版本与CUDA匹配")
        print("3. 尝试减小 batch_size 参数")
        print("4. 检查模型目录结构是否正确")
        print(f"5. 确认模型目录 {MODEL_PATH} 包含 saved_model.pb 和 variables 文件夹")