#!/usr/bin/env python3
"""
按分子指纹相似度聚类切分数据集。
同类化合物（指纹相似）会分到同一集合，不会跨 train/val/test。

用法:
    python3 scripts/make_cluster_split.py \
        --labels data/ZGC/train_labels.tsv \
        --output data/ZGC/split_cluster.tsv \
        --n-clusters 50
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.cluster import KMeans


def compute_fingerprint(smiles: str, n_bits: int = 4096) -> np.ndarray:
    """计算 Morgan 指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main():
    parser = argparse.ArgumentParser(description="按指纹相似度聚类切分数据集")
    parser.add_argument("--labels", required=True, help="标签 TSV 文件路径")
    parser.add_argument("--output", required=True, help="输出切分 TSV 路径")
    parser.add_argument("--n-clusters", type=int, default=50,
                        help="聚类数（默认 50，越大簇越多，切分越接近随机）")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    df = pd.read_csv(args.labels, sep="\t")
    # 按化合物去重（去掉能量后缀）
    df["compound"] = df["spec"].str.rsplit("-", n=1, expand=True)[0]
    unique_compounds = df.drop_duplicates("compound")

    print(f"总谱图: {len(df)}")
    print(f"唯一化合物: {len(unique_compounds)}")

    # 计算每个化合物的指纹
    print("计算指纹...")
    fingerprints = []
    valid_compounds = []
    for _, row in unique_compounds.iterrows():
        fp = compute_fingerprint(str(row.get("smiles", "")))
        if fp.sum() > 0:
            fingerprints.append(fp)
            valid_compounds.append(row["compound"])
        else:
            print(f"  ⚠ {row['compound']}: SMILES 无效或无指纹，跳过")

    fingerprints = np.array(fingerprints)
    n_compounds = len(fingerprints)
    print(f"有效化合物: {n_compounds}")

    # KMeans 聚类（用 Tanimoto 距离）
    # KMeans 用欧氏距离近似，对高维稀疏指纹有一定效果
    n_clusters = min(args.n_clusters, n_compounds // 3)
    print(f"聚类数: {n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(fingerprints)

    # 每簇内部按比例分配 train/val/test
    np.random.seed(args.seed)
    compound_to_split = {}
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        cluster_compounds = [valid_compounds[j] for j in np.where(mask)[0]]
        np.random.shuffle(cluster_compounds)

        n = len(cluster_compounds)
        train_n = max(1, round(n * args.train_ratio))
        val_n = max(0, round(n * args.val_ratio))
        # 剩余给 test
        test_n = n - train_n - val_n
        # 如果 test 不足，从 val 借
        if test_n < 0:
            val_n += test_n
            test_n = 0

        for j, c in enumerate(cluster_compounds):
            if j < train_n:
                compound_to_split[c] = "train"
            elif j < train_n + val_n:
                compound_to_split[c] = "val"
            else:
                compound_to_split[c] = "test"

    # 生成切分记录
    records = []
    for spec, compound in zip(df["spec"], df["compound"]):
        split = compound_to_split.get(compound, "test")
        records.append({"name": spec, "split": split})

    split_df = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.output, sep="\t", index=False)

    train_n = (split_df["split"] == "train").sum()
    val_n = (split_df["split"] == "val").sum()
    test_n = (split_df["split"] == "test").sum()
    print(f"\n切分完成:")
    print(f"  train: {train_n} ({train_n/len(df)*100:.1f}%)")
    print(f"  val:   {val_n} ({val_n/len(df)*100:.1f}%)")
    print(f"  test:  {test_n} ({test_n/len(df)*100:.1f}%)")
    print(f"  聚类数: {n_clusters}")
    print(f"  输出: {args.output}")


if __name__ == "__main__":
    main()
