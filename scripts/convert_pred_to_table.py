#!/usr/bin/env python3
"""
将 MIST 预测结果 .p 文件转换为表格文件（TSV）

输出:
    spec_name: 谱图名称
    其余列为 fp_0000 ~ fp_4095，每个值为 0~1 的概率

用法:
    python3 scripts/convert_pred_to_table.py \
        --pred results/ZGC/test/fp_preds/fp_preds_ZGC_test.p \
        --output results/ZGC/test/fp_preds/fp_preds_table.tsv
"""
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="将 .p 文件转换为表格 TSV")
    parser.add_argument("--pred", required=True, help="预测 .p 文件路径")
    parser.add_argument("--output", required=True, help="输出 TSV 路径")
    args = parser.parse_args()

    with open(args.pred, "rb") as f:
        data = pickle.load(f)

    names = data["names"]
    probs = data["preds"]  # (N, 4096)

    # 构建 DataFrame
    df = pd.DataFrame(probs, dtype=float)
    df.columns = [f"fp_{i:04d}" for i in range(probs.shape[1])]
    df.insert(0, "spec_name", names)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)

    print(f"✅ 已转换: {len(names)} 条谱图 × {probs.shape[1]} 位")
    print(f"   输出: {args.output}")


if __name__ == "__main__":
    main()
