#!/usr/bin/env python3
import argparse
import pandas as pd
from core.builder import VectorIndexBuilder
from core.model import load_models
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Build FAISS index')
    parser.add_argument('--item_data', type=str, required=True, help='Path to item data file')
    parser.add_argument('--output_dir', type=str, default='data/vectors', help='Output directory for index')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory containing trained models')
    args = parser.parse_args()

    try:
        # 加载数据
        logger.info(f"Loading item data from {args.item_data}")
        items = pd.read_parquet(args.item_data)

        # 加载模型
        logger.info("Loading item tower model")
        _, item_tower = load_models(args.model_dir)

        # 构建索引
        logger.info("Building index")
        builder = VectorIndexBuilder(item_tower)
        builder.build_index(items, args.output_dir)

        logger.info("Index build completed successfully")
    except Exception as e:
        logger.error(f"Index build failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()