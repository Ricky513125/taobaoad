# preprocess_sequences.py
import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
import pickle
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def generate_user_sequences(behavior_path, output_path, chunk_size=1_000_000):
    """内存优化的用户序列生成"""
    print(f"Generating user sequences from {behavior_path}...")

    # 第一次遍历：获取所有用户ID
    print("Scanning unique users...")
    chunks = pd.read_csv(behavior_path, chunksize=chunk_size, usecols=['user'])
    unique_users = set()
    for chunk in tqdm(chunks):
        unique_users.update(chunk['user'].unique())
    unique_users = sorted(unique_users)

    # 第二次遍历：流式处理每个用户
    print("Processing user sequences...")
    with open(output_path, 'w') as f_out:
        f_out.write("user,hist_sequence\n")

        for user in tqdm(unique_users):
            user_data = []
            for chunk in pd.read_csv(behavior_path, chunksize=chunk_size,
                                     dtype={'user': 'int32', 'cate': 'int32', 'brand': 'int32'}):
                chunk = chunk[chunk['user'] == user]
                if not chunk.empty:
                    user_data.append(chunk)

            if user_data:
                df_user = pd.concat(user_data)
                seq = '|'.join([
                    f"{row['cate']},{row['brand']}"
                    for _, row in df_user.sort_values('time_stamp').iterrows()
                ])
                f_out.write(f"{user},{seq}\n")

            del user_data
            gc.collect()

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

if __name__ == "__main__":
    # 配置路径
    os.makedirs("../data", exist_ok=True)
    behavior_path = "../data/cleaned_behavior.csv"
    user_seq_path = "../data/user_behavior_sequences.csv"
    item_seq_path = "../data/item_graph_sequences.csv"
    graph_path = "../data/item_graph.pkl"

    try:
        # 生成用户序列
        generate_user_sequences(behavior_path, user_seq_path)

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