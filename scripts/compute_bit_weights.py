#!/usr/bin/env python3
"""
从训练集 SMILES 计算指纹每位出现频率，生成频率加权损失权重。
低频位权重高 → 模型更关注稀有子结构 → 有助于提升精确匹配。
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def main():
    parser = argparse.ArgumentParser(description="计算指纹位频率权重")
    parser.add_argument("--labels", required=True, help="训练集 labels.tsv（需含 SMILES 列）")
    parser.add_argument("--output", required=True, help="输出权重 .npy 文件")
    parser.add_argument("--n-bits", type=int, default=4096)
    parser.add_argument("--power", type=float, default=1.0,
                        help="权重缩放指数")
    parser.add_argument("--low-freq-weight", type=float, default=None,
                        help="低频位权重（频率<1%）")
    parser.add_argument("--mid-freq-weight", type=float, default=None,
                        help="中频位权重（频率1%~20%）")
    parser.add_argument("--high-freq-weight", type=float, default=None,
                        help="高频位权重（频率>20%）")
    args = parser.parse_args()

    df = pd.read_csv(args.labels, sep="\t")

    # 计算每个化合物的指纹
    bit_counts = np.zeros(args.n_bits)
    n_valid = 0
    for smi in df["smiles"].dropna().unique():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=args.n_bits)
        arr = np.zeros(args.n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        bit_counts += arr
        n_valid += 1

    # 频率
    freq = bit_counts / n_valid  # 0~1

    # 权重计算
    weights = np.ones(args.n_bits)

    if args.low_freq_weight is not None:
        # 分段权重模式
        weights[freq <= 0.01] = args.low_freq_weight
        weights[(freq > 0.01) & (freq <= 0.05)] = 1.0
        weights[(freq > 0.05) & (freq <= 0.20)] = args.mid_freq_weight or 1.5
        weights[freq > 0.20] = args.high_freq_weight or 3.0
    else:
        # 原始频率公式模式（高频优先）
        weights = (freq + 0.01) ** args.power
        weights = weights / weights.mean()

    np.save(args.output, weights)

    # 打印统计
    print(f"有效化合物: {n_valid}")
    print(f"平均每位频率: {freq.mean():.4f}")
    print(f"最高频位: {freq.max():.4f}")
    print(f"最低频位: {freq.min():.4f}")
    print(f"权重范围: [{weights.min():.2f}, {weights.max():.2f}]")
    # 按频率分组
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    for i in range(len(bins)-1):
        mask = (freq > bins[i]) & (freq <= bins[i+1])
        if mask.sum() > 0:
            print(f"  频率 {bins[i]:.2f}~{bins[i+1]:.2f}: {mask.sum():>4} 位, "
                  f"平均权重 {weights[mask].mean():.2f}")

    print(f"\n输出: {args.output}")


if __name__ == "__main__":
    main()
