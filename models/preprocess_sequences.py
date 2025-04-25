# preprocess_sequences.py
import pandas as pd
import numpy as np
from graph_argument import build_item_graph, generate_sequences
import pickle
import os
from collections import defaultdict
from tqdm import tqdm
import networkx as nx


def generate_user_sequences(behavior_path, output_path):
    """生成用户历史行为序列"""
    df = pd.read_csv(behavior_path)

    # 按用户分组生成历史行为序列 (格式: cate1,brand1|cate2,brand2|...)
    user_sequences = df.groupby('user').apply(
        lambda x: '|'.join([
            f"{row['cate']},{row['brand']}"
            for _, row in x.sort_values('time_stamp').iterrows()
        ])
    )

    user_sequences.to_csv(output_path, header=True, index=True)
    print(f"Saved user sequences to {output_path}")
    return user_sequences


def generate_item_sequences(behavior_path, output_path, walks_per_node=5):
    """生成物品图游走序列"""
    # 1. 构建物品共现图
    print("Building item graph...")
    df = pd.read_csv(behavior_path)

    # 创建物品节点 (使用cate和brand组合作为节点ID)
    df['item_node'] = df['cate'].astype(str) + '_' + df['brand'].astype(str)

    # 统计共现关系
    co_occur = defaultdict(lambda: defaultdict(int))
    for _, group in df.groupby('user'):
        items = group.sort_values('time_stamp')['item_node'].values
        for i in range(len(items) - 1):
            u, v = items[i], items[i + 1]
            co_occur[(u, v)] += 1

    # 构建有向加权图
    G = nx.DiGraph()
    for (u, v), weight in co_occur.items():
        G.add_edge(u, v, weight=weight)

    # 2. 为每个物品生成随机游走序列
    print("Generating item sequences...")
    all_sequences = []
    nodes = list(G.nodes())

    for node in nodes:
        for _ in range(walks_per_node):
            walk = [node]
            current_node = node

            for _ in range(10):  # 游走长度
                neighbors = list(G.successors(current_node))
                if not neighbors:
                    break

                weights = [G[current_node][n]['weight'] for n in neighbors]
                next_node = np.random.choice(neighbors, p=weights / np.sum(weights))
                walk.append(next_node)
                current_node = next_node

            all_sequences.append((node, ' '.join(walk)))

    # 保存序列数据
    item_sequences = pd.DataFrame(all_sequences, columns=['item', 'graph_seq'])
    item_sequences.to_csv(output_path, index=False)
    print(f"Saved item sequences to {output_path}")
    return item_sequences


if __name__ == "__main__":
    # 配置路径
    behavior_path = "data/cleaned_behavior.csv"
    user_seq_path = "data/user_behavior_sequences.csv"
    item_seq_path = "data/item_graph_sequences.csv"

    # 生成序列数据
    print("Generating user behavior sequences...")
    generate_user_sequences(behavior_path, user_seq_path)

    print("\nGenerating item graph sequences...")
    generate_item_sequences(behavior_path, item_seq_path)

    # 保存物品关系图 (可选)
    # with open("data/item_graph.pkl", 'wb') as f:
    #     pickle.dump(G, f)