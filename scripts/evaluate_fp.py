#!/usr/bin/env python3
"""
评估 MIST 指纹预测命中率

用法:
    python scripts/evaluate_fp.py \
        --pred results/ZGC/test/fp_preds_table.tsv \
        --true data/ZGC/test_molecular_fingerprints.tsv \
        --output results/ZGC/test/hit_rate.tsv
"""
import argparse
import re
from pathlib import Path
import pandas as pd


def parse_fp_to_set(fp_string):
    return set(re.findall(r"<fp\d{4}>", fp_string))


def calc_tanimoto(s1, s2):
    set1 = parse_fp_to_set(s1)
    set2 = parse_fp_to_set(s2)
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def classify_similarity(score):
    if score >= 0.85:
        return "高相似"
    elif score >= 0.5:
        return "中等相似"
    else:
        return "低相似"


def main():
    parser = argparse.ArgumentParser(description="评估 MIST 指纹预测命中率")
    parser.add_argument("--pred", required=True, help="预测指纹 TSV")
    parser.add_argument("--true", required=True, help="真值指纹 TSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="二值化阈值")
    parser.add_argument("--output", default=None, help="结果保存路径")
    args = parser.parse_args()

    # 加载预测（你的格式：spec_name + fp_0000~fp_4095 概率值）
    data1 = pd.read_csv(args.pred, sep="\t")
    fp_cols = [c for c in data1.columns if c.startswith("fp_")]

    # 提取 ID 并二值化转 <fpNNNN>
    def to_sparse(row):
        bits = [i for i, c in enumerate(fp_cols) if float(row[c]) >= args.threshold]
        return "".join(f"<fp{i:04d}>" for i in bits)

    data1["fp_sequence"] = data1.apply(to_sparse, axis=1)
    data1["ID"] = data1["spec_name"].apply(lambda x: x.split("-")[0])

    # 加载真值
    data2 = pd.read_csv(args.true, sep="\t")

    # 合并
    data = pd.merge(data1, data2, on="ID")

    # 完全匹配
    exact = int((data["fp_sequence"] == data["FingerPrints"]).sum())
    total = len(data)

    # Tanimoto
    data["tanimoto"] = data.apply(
        lambda r: calc_tanimoto(r["fp_sequence"], r["FingerPrints"]), axis=1
    )
    data["相似度等级"] = data["tanimoto"].apply(classify_similarity)

    # 输出
    print("=" * 50)
    print("Tanimoto相似度统计")
    print("=" * 50)
    print(f"完全匹配: {exact}/{total} ({exact/total*100:.2f}%)")
    print(f"平均Tanimoto: {data['tanimoto'].mean():.4f}")
    print(f"最小Tanimoto: {data['tanimoto'].min():.4f}")
    print(f"最大Tanimoto: {data['tanimoto'].max():.4f}")
    print()

    print("=" * 50)
    print("相似度等级分布")
    print("=" * 50)
    for grade in ["高相似", "中等相似", "低相似"]:
        cnt = (data["相似度等级"] == grade).sum()
        print(f"{grade}: {cnt} 条 ({cnt/total*100:.2f}%)")

    print("\n前10行结果:")
    print(data[["ID", "tanimoto", "相似度等级"]].head(10).to_string(index=False))

    if args.output:
        out = data[["ID", "fp_sequence", "FingerPrints", "tanimoto", "相似度等级"]]
        out.to_csv(args.output, sep="\t", index=False)
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
