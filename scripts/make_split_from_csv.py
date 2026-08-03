#!/usr/bin/env python3
"""
从已有的 train/val/test CSV 文件生成 MIST 切分文件。

用法:
    python3 scripts/make_split_from_csv.py \
        --train data/cluster/train.csv \
        --val data/cluster/val.csv \
        --test data/cluster/test.csv \
        --output data/cluster/split.tsv
"""
import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="从 CSV 生成切分文件")
    parser.add_argument("--train", required=True, help="训练集 CSV")
    parser.add_argument("--val", required=True, help="验证集 CSV")
    parser.add_argument("--test", required=True, help="测试集 CSV")
    parser.add_argument("--output", required=True, help="输出 split.tsv 路径")
    parser.add_argument("--id-col", default="ID", help="CSV 中 ID 列名（默认 ID）")
    args = parser.parse_args()

    train = pd.read_csv(args.train, encoding="utf-8-sig")
    val = pd.read_csv(args.val, encoding="utf-8-sig")
    test = pd.read_csv(args.test, encoding="utf-8-sig")

    records = []
    # 从 labels.tsv 取完整谱图名（带 _20 后缀）
    labels_train = pd.read_csv(args.train.replace(".csv", "_labels.tsv"), sep="\t")
    labels_val = pd.read_csv(args.val.replace(".csv", "_labels.tsv"), sep="\t")
    labels_test = pd.read_csv(args.test.replace(".csv", "_labels.tsv"), sep="\t")

    for name in labels_train["spec"]:
        records.append({"name": name, "split": "train"})
    for name in labels_val["spec"]:
        records.append({"name": name, "split": "val"})
    for name in labels_test["spec"]:
        records.append({"name": name, "split": "test"})

    split_df = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.output, sep="\t", index=False)

    train_n = (split_df["split"] == "train").sum()
    val_n = (split_df["split"] == "val").sum()
    test_n = (split_df["split"] == "test").sum()
    print(f"输出: {args.output}")
    print(f"  train: {train_n}")
    print(f"  val:   {val_n}")
    print(f"  test:  {test_n}")


if __name__ == "__main__":
    main()
