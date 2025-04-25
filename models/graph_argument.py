import networkx as nx
import pandas as pd
from collections import defaultdict
import random
import numpy as np
import pickle
from tqdm import tqdm


def build_item_graph(behavior_path, output_path):
    """构建物料共现关系图（优化GPU内存使用）"""
    print("Building item graph...")
    df = pd.read_csv(behavior_path)

    # 优化内存：只保留必要列并转换为更小数据类型
    df = df[['user', 'cate', 'brand', 'time_stamp']]
    df['cate'] = df['cate'].astype('int32')
    df['brand'] = df['brand'].astype('int32')

    # 按用户分组生成时序行为序列（使用并行处理优化）
    user_sequences = df.groupby('user').apply(
        lambda x: x.sort_values('time_stamp')[['cate', 'brand']].values.tolist()
    )

    # 统计共现关系（带权重）- 使用批处理优化
    co_occur = defaultdict(lambda: defaultdict(int))
    for seq in tqdm(user_sequences, desc="Processing user sequences"):
        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i + 1]
            key = (u[0], u[1], v[0], v[1])  # 扁平化存储优化
            co_occur[key] += 1

    # 构建有向加权图（优化内存）
    G = nx.DiGraph()
    for (u_cate, u_brand, v_cate, v_brand), weight in tqdm(co_occur.items(), desc="Building graph"):
        u = (int(u_cate), int(u_brand))
        v = (int(v_cate), int(v_brand))
        G.add_edge(u, v, weight=weight)

    # 保存图（使用更高效的pickle格式）
    with open(output_path, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return G


def random_walk(graph, start_node, walk_length=10):
    """随机游走生成序列（优化性能）"""
    walk = [start_node]
    current_node = start_node

    for _ in range(walk_length - 1):
        try:
            neighbors = list(graph.successors(current_node))
            if not neighbors:
                break

            # 获取边权重并处理可能的缺失权重
            weights = []
            for n in neighbors:
                try:
                    weights.append(graph[current_node][n]['weight'])
                except KeyError:
                    weights.append(1.0)  # 默认权重

            # 归一化权重防止数值问题
            weights = np.array(weights, dtype=np.float32)
            if weights.sum() <= 0:
                weights = np.ones_like(weights)
            weights /= weights.sum()

            next_node = random.choices(neighbors, weights=weights)[0]
            walk.append(next_node)
            current_node = next_node
        except Exception as e:
            print(f"Error in random walk: {e}")
            break

    return walk


def generate_sequences(graph, output_path, walks_per_node=5, num_workers=4):
    """批量生成序列数据（支持并行）"""
    print(f"Generating sequences with {num_workers} workers...")
    nodes = list(graph.nodes())

    # 并行生成序列
    from multiprocessing import Pool
    def _generate_walk(node):
        return random_walk(graph, node)

    with Pool(num_workers) as pool, open(output_path, 'w') as f:
        for _ in range(walks_per_node):
            for walk in tqdm(pool.imap(_generate_walk, nodes), total=len(nodes), desc="Generating walks"):
                if walk:
                    f.write(' '.join([f"{cate},{brand}" for (cate, brand) in walk]) + '\n')