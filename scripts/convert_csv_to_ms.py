#!/usr/bin/env python3
"""
将 data/cluster 的 CSV 格式转换为 MIST 推理所需的 .ms 文件 + labels.tsv

CSV 列: ID, Name, Formula, CASId, InChIKey, SMILES,
         ScanFilter, RetentionTime, PrecursorMass,
         mz1~mz100, Intensity1~Intensity100

用法:
    python3 scripts/convert_csv_to_ms.py \
        --csv data/cluster/test.csv \
        --output-dir data/cluster_ms/test \
        --labels data/cluster/labels.tsv
"""
import argparse
import csv
from pathlib import Path


def convert_csv(csv_path, output_dir, labels_path, dataset_name):
    """CSV → .ms 文件 + labels.tsv"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = open(csv_path, "r", encoding="utf-8-sig").readlines()
    reader = csv.DictReader(lines, delimiter=",")

    label_rows = []
    count = 0
    for row in reader:
        spec_name = row.get("ID") or row.get("Name") or f"spec_{count}"
        formula = row["Formula"]
        parentmass = row["PrecursorMass"]
        smiles = row.get("SMILES", "")
        inchikey = row.get("InChIKey", "")
        # CSV 没有电离方式，默认 [M+H]+
        ionization = "[M+H]+"

        # 收集 m/z 和强度对
        peaks = []
        for i in range(1, 101):
            mz_key = f"mz{i}"
            inten_key = f"Intensity{i}"
            mz = row.get(mz_key, "").strip()
            inten = row.get(inten_key, "").strip()
            if mz and inten and float(inten) > 0:
                peaks.append((float(mz), float(inten)))

        if not peaks:
            print(f"  ⚠ {spec_name}: 无有效峰，跳过")
            continue

        # 写入 .ms 文件
        ms_path = output_dir / f"{spec_name}.ms"
        with open(ms_path, "w") as f:
            f.write(f">compound {spec_name}\n")
            f.write(f">formula {formula}\n")
            f.write(f">parentmass {parentmass}\n")
            f.write(f">ionization {ionization}\n")
            if smiles:
                f.write(f">smiles {smiles}\n")
            if inchikey:
                f.write(f">InChIKey {inchikey}\n")
            f.write(f"\n>ms2peaks\n")
            for mz, inten in peaks:
                f.write(f"{mz:.4f} {inten:.6f}\n")

        # 标签
        label_rows.append({
            "spec": spec_name,
            "formula": formula,
            "ionization": ionization,
            "dataset": dataset_name,
            "compound": spec_name,
            "parentmass": parentmass,
            "instrument": "unknown",
            "smiles": smiles,
            "inchikey": inchikey,
        })
        count += 1

    # 写入 labels.tsv
    Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        cols = ["spec", "formula", "ionization", "dataset", "compound",
                "parentmass", "instrument", "smiles", "inchikey"]
        f.write("\t".join(cols) + "\n")
        for r in label_rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    print(f"✅ {csv_path}")
    print(f"   .ms 文件: {count} 个 → {output_dir}/")
    print(f"   标签文件: {labels_path}")


def main():
    parser = argparse.ArgumentParser(description="CSV → .ms + labels.tsv")
    parser.add_argument("--csv", required=True, help="输入的 CSV 文件")
    parser.add_argument("--output-dir", required=True, help="输出的 .ms 目录")
    parser.add_argument("--labels", required=True, help="输出的 labels.tsv 路径")
    parser.add_argument("--dataset", default="cluster", help="数据集名称")
    args = parser.parse_args()

    convert_csv(args.csv, args.output_dir, args.labels, args.dataset)


if __name__ == "__main__":
    main()
