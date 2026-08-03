#!/usr/bin/env python3
"""
从标签文件中生成 train/val 切分文件

用法:
    python3 scripts/make_split.py \
        --labels data/ZGC/train_labels.tsv \
        --output data/ZGC/split.tsv \
        --train-ratio 0.8 --seed 42
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="生成数据集切分文件")
    parser.add_argument("--labels", required=True, help="标签 TSV 文件路径")
    parser.add_argument("--output", required=True, help="输出切分 TSV 路径")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例（默认 0.8）")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例（默认 0.1），剩余为测试集")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    df = pd.read_csv(args.labels, sep="\t")
    # 按化合物 ID 去重（去掉后缀）
    df["compound"] = df["spec"].str.rsplit("-", n=1, expand=True)[0]
    compounds = df["compound"].unique()
    np.random.seed(args.seed)
    np.random.shuffle(compounds)

    n = len(compounds)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)

    train_compounds = set(compounds[:train_end])
    val_compounds = set(compounds[train_end:val_end])
    test_compounds = set(compounds[val_end:])

    records = []
    for spec, compound in zip(df["spec"], df["compound"]):
        if compound in train_compounds:
            records.append({"name": spec, "split": "train"})
        elif compound in val_compounds:
            records.append({"name": spec, "split": "val"})
        else:
            records.append({"name": spec, "split": "test"})

    split_df = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.output, sep="\t", index=False)
    train_n = (split_df["split"] == "train").sum()
    val_n = (split_df["split"] == "val").sum()
    test_n = (split_df["split"] == "test").sum()
    print(f"切分完成: train={train_n}, val={val_n}, test={test_n}")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
