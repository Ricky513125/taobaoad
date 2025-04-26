import os
import faiss
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import time
import gc
from multiprocessing import cpu_count
from keras.layers import TFSMLayer

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
    """GPU优化的模型加载 - 适配Keras 3"""
    try:
        # 显式指定GPU设备
        with tf.device('/GPU:0'):
            # 尝试直接加载Keras 3格式
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'SqueezeLayer': SqueezeLayer}
                )
                return model
            except:
                # 如果失败，尝试使用TFSMLayer加载旧格式
                print("检测到旧模型格式，使用TFSMLayer加载...")
                inputs = {
                    'item_adgroup_id': tf.keras.Input(shape=(1,), dtype=tf.int32, name='item_adgroup_id'),
                    'item_cate_id': tf.keras.Input(shape=(1,), dtype=tf.int32, name='item_cate_id'),
                    'item_campaign_id': tf.keras.Input(shape=(1,), dtype=tf.int32, name='item_campaign_id'),
                    'item_customer': tf.keras.Input(shape=(1,), dtype=tf.int32, name='item_customer'),
                    'item_brand': tf.keras.Input(shape=(1,), dtype=tf.float32, name='item_brand'),
                    'item_price': tf.keras.Input(shape=(1,), dtype=tf.float32, name='item_price')
                }

                # 使用TFSMLayer加载旧模型
                sm_layer = TFSMLayer(
                    model_path,
                    call_endpoint='serving_default',
                    name='loaded_model'
                )
                outputs = sm_layer(inputs)

                # 构建新模型
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                return model
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}\n建议: 1. 检查模型路径 2. 确认模型格式 3. 尝试转换模型为.keras格式")


# 其余函数保持不变...
def create_dataset(item_data, batch_size=1024):
    """创建TF Dataset管道（GPU优化）"""
    # ... (保持原样)


def generate_vectors_gpu(model, item_data, batch_size=8192):
    """GPU加速的向量生成"""
    # ... (保持原样)


def build_faiss_index(vectors, ids, output_dir="recall"):
    """GPU支持的索引构建"""
    # ... (保持原样)


if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "models/item_tower"
    ITEM_DATA_PATH = "data/ad_feature.csv"

    try:
        # 1. 加载数据
        print("加载物料数据...")
        item_data = pd.read_csv(ITEM_DATA_PATH)
        print(f"数据量: {len(item_data):,} 条")

        # 2. 加载模型 - 现在支持Keras 3和旧格式
        print("\n加载GPU优化模型...")
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
        print("4. 检查模型格式是否为Keras 3支持的.keras或.h5格式")
        print("5. 考虑使用 tf.keras.models.save_model(model, 'model.keras') 转换模型格式")