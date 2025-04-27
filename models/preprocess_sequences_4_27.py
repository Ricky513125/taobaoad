# preprocess_sequences.py
import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
import pickle
import time
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# def generate_user_sequences(behavior_path, output_path, chunk_size=1_000_000):
#     """内存优化的用户序列生成"""
#     print(f"Generating user sequences from {behavior_path}...")
#
#     # 第一次遍历：获取所有用户ID
#     print("Scanning unique users...")
#     chunks = pd.read_csv(behavior_path, chunksize=chunk_size, usecols=['user'])
#     unique_users = set()
#     for chunk in tqdm(chunks):
#         unique_users.update(chunk['user'].unique())
#     unique_users = sorted(unique_users)
#
#     # 第二次遍历：流式处理每个用户
#     print("Processing user sequences...")
#     with open(output_path, 'w') as f_out:
#         f_out.write("user,hist_sequence\n")
#
#         for user in tqdm(unique_users):
#             user_data = []
#             for chunk in pd.read_csv(behavior_path, chunksize=chunk_size,
#                                      dtype={'user': 'int32', 'cate': 'int32', 'brand': 'int32'}):
#                 chunk = chunk[chunk['user'] == user]
#                 if not chunk.empty:
#                     user_data.append(chunk)
#
#             if user_data:
#                 df_user = pd.concat(user_data)
#                 seq = '|'.join([
#                     f"{row['cate']},{row['brand']}"
#                     for _, row in df_user.sort_values('time_stamp').iterrows()
#                 ])
#                 f_out.write(f"{user},{seq}\n")
#
#             del user_data
#             gc.collect()
#
#     print(f"Saved user sequences to {output_path}")

def generate_user_sequences_optimized(behavior_path, output_path, chunk_size=1_000_000):
    """内存和IO优化的用户序列生成"""
    print(f"Generating user sequences from {behavior_path}...")

    # 第一次遍历：获取所有用户ID
    print("Scanning unique users...")
    chunks = pd.read_csv(behavior_path, chunksize=chunk_size, usecols=['user'])
    unique_users = set()
    for chunk in tqdm(chunks):
        unique_users.update(chunk['user'].unique())
    unique_users = sorted(unique_users)

    # 创建用户序列字典
    user_sequences = defaultdict(list)

    # 第二次遍历：单次流式处理构建所有用户序列
    print("Processing user sequences...")
    chunks = pd.read_csv(behavior_path, chunksize=chunk_size,
                         dtype={'user': 'int32', 'cate': 'int32', 'brand': 'int32', 'time_stamp': 'int32'})

    for chunk in tqdm(chunks, desc="Processing chunks"):
        # 按用户分组并排序
        grouped = chunk.sort_values(['user', 'time_stamp']).groupby('user')
        for user, group in grouped:
            seq = '|'.join(f"{row['cate']},{row['brand']}" for _, row in group.iterrows())
            user_sequences[user].append(seq)

    # 写入文件
    print("Writing output...")
    with open(output_path, 'w') as f_out:
        f_out.write("user,hist_sequence\n")
        for user in tqdm(unique_users, desc="Writing users"):
            if user in user_sequences:
                full_seq = '|'.join(user_sequences[user])
                f_out.write(f"{user},{full_seq}\n")

    print(f"Saved user sequences to {output_path}")

def generate_item_sequences(behavior_path, output_path, walks_per_node=5, use_gpu=False):
    """改进版物品序列生成（支持GPU加速）"""
    print("Building item graph...")

    # 1. 构建共现图（CPU）
    co_occur = defaultdict(lambda: defaultdict(int))
    chunks = pd.read_csv(behavior_path, chunksize=1_000_000,
                         dtype={'user': 'int32', 'cate': 'int32', 'brand': 'int32'})

    for chunk in tqdm(chunks, desc="Processing chunks"):
        for _, group in chunk.groupby('user'):
            items = group.sort_values('time_stamp').apply(
                lambda x: f"{x['cate']}_{x['brand']}", axis=1).values
            for i in range(len(items) - 1):
                co_occur[(items[i], items[i + 1])] += 1

    G = nx.DiGraph()
    for (u, v), weight in tqdm(co_occur.items(), desc="Building graph"):
        G.add_edge(u, v, weight=weight)

    # 2. 序列生成（可选GPU）
    print(f"Generating random walks (GPU: {use_gpu})...")
    nodes = list(G.nodes())

    if use_gpu and GPU_AVAILABLE:
        # GPU加速版本
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        weight_matrix = cp.zeros((len(nodes), len(nodes)), dtype='float32')

        for u, v, d in G.edges(data=True):
            weight_matrix[node_to_idx[u], node_to_idx[v]] = d['weight']

        with open(output_path, 'w') as f_out:
            f_out.write("item,graph_seq\n")
            for i in tqdm(range(len(nodes)), desc="GPU Random Walks"):
                walk = [nodes[i]]
                current = i
                for _ in range(10):
                    neighbors = cp.where(weight_matrix[current] > 0)[0]
                    if len(neighbors) == 0:
                        break
                    probs = weight_matrix[current, neighbors] / cp.sum(weight_matrix[current, neighbors])
                    current = cp.random.choice(neighbors, p=probs.get())
                    walk.append(nodes[current])
                f_out.write(f"{nodes[i]},{' '.join(walk)}\n")
    else:
        # CPU版本（原逻辑）
        with open(output_path, 'w') as f_out:
            f_out.write("item,graph_seq\n")
            for node in tqdm(nodes, desc="Nodes processed"):
                for _ in range(walks_per_node):
                    walk = [node]
                    current = node
                    for _ in range(10):
                        neighbors = list(G.successors(current))
                        if not neighbors:
                            break
                        weights = np.array([G[current][n]['weight'] for n in neighbors])
                        prob = weights / weights.sum()
                        next_node = np.random.choice(neighbors, p=prob)
                        walk.append(next_node)
                        current = next_node
                    f_out.write(f"{node},{' '.join(walk)}\n")

    return G


def generate_user_sequences_fast(behavior_path, output_path):
    print("Loading data to memory...")
    df = pd.read_csv(
        behavior_path,
        dtype={'user': 'int32', 'cate': 'int16', 'brand': 'int16', 'time_stamp': 'int32'}
    )

    print("Sorting and grouping...")
    df = df.sort_values(['user', 'time_stamp'])
    df['item_pair'] = df['cate'].astype(str) + ',' + df['brand'].astype(str)
    sequences = df.groupby('user')['item_pair'].agg('|'.join).reset_index()

    print("Saving...")
    sequences.to_csv(output_path, index=False, header=['user', 'hist_sequence'])
    print(f"Done! Saved to {output_path}")


def generate_user_sequences_streaming(behavior_path, output_path, chunk_size=50_000):
    print("Processing in chunks...")

    # 初始化进度条
    pbar_total = None
    start_time = time.time()

    # 第一次遍历：收集所有用户（带进度条）
    print("🔍 Scanning all users...")
    chunks = pd.read_csv(behavior_path, chunksize=chunk_size, usecols=['user'])
    unique_users = set()

    # 先计算总chunk数用于进度条
    with open(behavior_path, 'r') as f:
        total_chunks = sum(1 for _ in f) // chunk_size + 1

    with tqdm(total=total_chunks, desc="Scanning chunks") as pbar:
        for chunk in chunks:
            unique_users.update(chunk['user'].unique())
            pbar.update(1)

    unique_users = sorted(unique_users)
    user_count = len(unique_users)
    print(f"✅ Found {user_count} unique users")

    # 第二次遍历：处理用户（带动态预估）
    writer = open(output_path, 'w')
    writer.write("user,hist_sequence\n")

    print("🔄 Processing user sequences...")
    with tqdm(total=user_count, desc="Users processed", unit='user',
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [ETA:{remaining}]") as user_pbar:

        for i, user in enumerate(unique_users):
            user_start_time = time.time()
            chunks = pd.read_csv(
                behavior_path,
                chunksize=chunk_size,
                dtype={'user': 'int32', 'cate': 'int16', 'brand': 'int16', 'time_stamp': 'int32'}
            )
            user_sequences = []

            for chunk in chunks:
                chunk = chunk[chunk['user'] == user]
                if not chunk.empty:
                    chunk_sorted = chunk.sort_values('time_stamp')
                    seq = '|'.join(f"{row['cate']},{row['brand']}" for _, row in chunk_sorted.iterrows())
                    user_sequences.append(seq)

            if user_sequences:
                writer.write(f"{user},{'|'.join(user_sequences)}\n")

            # 动态更新进度条信息
            user_pbar.set_postfix({
                'Speed': f"{1 / (time.time() - user_start_time):.1f} users/s",
                'Current User': user
            })
            user_pbar.update(1)


if __name__ == "__main__":
    # 配置路径
    os.makedirs("../data", exist_ok=True)
    behavior_path = "../data/cleaned_behavior.csv"
    user_seq_path = "../data/user_behavior_sequences.csv"
    item_seq_path = "../data/item_graph_sequences.csv"
    graph_path = "../data/item_graph.pkl"

    try:
        # 生成用户序列
        generate_user_sequences_streaming(behavior_path, user_seq_path)

        # 生成物品序列和图
        G = generate_item_sequences(behavior_path, item_seq_path,use_gpu=GPU_AVAILABLE)

        # 保存图
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f)
        print(f"Saved item graph to {graph_path}")

    except MemoryError:
        print("\nERROR: Memory exhausted. Try these solutions:")
        print("1. Reduce chunk_size in the function calls")
        print("2. Use smaller input data")
        print("3. Add swap space: sudo fallocate -l 20G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile")
        print("4. Run on a machine with more RAM")
    except Exception as e:
        print(f"\nERROR: {str(e)}")