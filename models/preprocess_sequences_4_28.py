# preprocess_sequences_optimized.py
import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
import networkx as nx
from collections import defaultdict, Counter
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def generate_user_sequences_optimized(behavior_path, output_path, chunk_size=2_000_000):
    """优化版用户行为序列生成（单次遍历+内存优化）"""
    print(f"🚀 开始处理用户行为序列: {behavior_path}")

    # 配置数据类型
    dtype = {
        'user': 'int32',
        'cate': 'int16',
        'brand': 'int16',
        'time_stamp': 'int32'
    }

    # 初始化数据结构
    user_sequences = defaultdict(list)

    # 带进度条的块处理
    with tqdm(desc="处理数据块", unit='chunk') as pbar:
        for chunk in pd.read_csv(behavior_path, chunksize=chunk_size, dtype=dtype):
            # 预计算组合键并排序
            chunk['item_pair'] = chunk['cate'].astype(str) + ',' + chunk['brand'].astype(str)
            sorted_chunk = chunk.sort_values(['user', 'time_stamp'])

            # 分组聚合
            grouped = sorted_chunk.groupby('user')['item_pair'].agg('|'.join)

            # 合并结果
            for user, seq in grouped.items():
                user_sequences[user].append(seq)

            # 内存清理
            del chunk, sorted_chunk, grouped
            gc.collect()
            pbar.update(1)

    # 写入最终结果
    print("🛠️ 合并用户序列...")
    with open(output_path, 'w') as f_out:
        f_out.write("user,hist_sequence\n")
        for user, parts in tqdm(user_sequences.items(), desc="写入文件"):
            full_seq = '|'.join(parts)
            f_out.write(f"{user},{full_seq}\n")

    print(f"✅ 用户序列保存至 {output_path} (共 {len(user_sequences)} 用户)")


def build_cooccurrence_graph(behavior_path, chunk_size=1_000_000):
    """高效构建共现图"""
    print("🕸️ 构建共现关系图...")
    co_occur = defaultdict(Counter)

    # 计算总块数
    with open(behavior_path, 'r') as f:
        total_chunks = sum(1 for _ in f) // chunk_size + 1

    # 分块处理
    with tqdm(total=total_chunks, desc="处理数据块") as pbar:
        for chunk in pd.read_csv(
                behavior_path,
                chunksize=chunk_size,
                dtype={'user': 'int32', 'cate': 'int16', 'brand': 'int16'}
        ):
            # 生成物品标识
            chunk['item'] = chunk['cate'].astype(str) + '_' + chunk['brand'].astype(str)

            # 处理用户序列
            for _, group in chunk.groupby('user'):
                items = group.sort_values('time_stamp')['item'].values
                for i in range(len(items) - 1):
                    co_occur[items[i]].update([items[i + 1]])

            pbar.update(1)
            del chunk
            gc.collect()

    return co_occur


def generate_item_sequences_optimized(behavior_path, output_path, walks_per_node=5):
    """并行优化版物品序列生成"""
    # 1. 构建共现图
    co_occur = build_cooccurrence_graph(behavior_path)

    # 2. 构建图结构
    print("🏗️ 创建图网络...")
    G = nx.DiGraph()
    for src, counters in tqdm(co_occur.items(), desc="添加边"):
        total = sum(counters.values())
        for dst, count in counters.items():
            G.add_edge(src, dst, weight=count / total)

    # 3. 预计算转移概率
    print("⚡ 预计算转移矩阵...")
    transition_probs = {}
    nodes = list(G.nodes())
    for node in tqdm(nodes, desc="处理节点"):
        neighbors = list(G.successors(node))
        if neighbors:
            probs = [G[node][n]['weight'] for n in neighbors]
            transition_probs[node] = (neighbors, np.array(probs))

    # 4. 并行生成游走序列
    print(f"🚶 生成随机游走 (并行 workers={os.cpu_count()})...")
    with open(output_path, 'w') as f_out:
        f_out.write("item,graph_seq\n")

        def generate_walks(node):
            walks = []
            for _ in range(walks_per_node):
                walk = [node]
                current = node
                for _ in range(10):
                    if current not in transition_probs:
                        break
                    neighbors, probs = transition_probs[current]
                    next_node = np.random.choice(neighbors, p=probs)
                    walk.append(next_node)
                    current = next_node
                return f"{node},{' '.join(walk)}\n"

        # 并行处理
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(generate_walks, node) for node in nodes]
            for future in tqdm(as_completed(futures), total=len(nodes), desc="生成序列"):
                f_out.write(future.result())

    return G


def save_graph(graph, output_path):
    """保存图对象"""
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)
    print(f"✅ 图结构已保存至 {output_path}")


def memory_guard(func):
    """内存保护装饰器"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MemoryError:
            print("\n⚠️ 内存不足！建议解决方案：")
            print("1. 减小 chunk_size 参数")
            print("2. 使用更高内存的机器")
            print("3. 增加交换空间（swap space）")
            print("4. 使用更小的数据类型（如int8）")
            exit(1)

    return wrapper


@memory_guard
def main():
    # 配置路径
    os.makedirs("../data", exist_ok=True)
    behavior_path = "../data/cleaned_behavior.csv"
    user_seq_path = "../data/sequence/user_sequences_optimized.csv"
    item_seq_path = "../data/sequence/item_sequences_optimized.csv"
    graph_path = "../data/item_graph_optimized.pkl"

    # 执行处理流程
    start_time = time.time()

    # 用户序列生成
    generate_user_sequences_optimized(behavior_path, user_seq_path)

    # 物品序列生成
    item_graph = generate_item_sequences_optimized(behavior_path, item_seq_path)
    save_graph(item_graph, graph_path)

    # 性能报告
    total_time = time.time() - start_time
    print(f"\n🎉 处理完成！总用时: {total_time // 3600:.0f}h {total_time % 3600 // 60:.0f}m {total_time % 60:.2f}s")

    # 内存清理
    gc.collect()


if __name__ == "__main__":
    main()