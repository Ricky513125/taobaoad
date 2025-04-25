"""
4.25 4:42 元宝套餐

"""
def predict_rank_score(user_data, item_data_list):
    """输入用户特征和候选物品列表，返回精排分数"""
    # 1. 生成精排特征
    rank_features = [
        build_rank_features_single(user_data, item_data)
        for item_data in item_data_list
    ]

    # 2. 预测CTR
    scores = model.predict(rank_features)

    # 3. 排序返回
    ranked_items = sorted(zip(item_data_list, scores), key=lambda x: -x[1])
    return ranked_items