#!/usr/bin/env python3
"""
从 .ms 文件头生成 MIST 所需的标签文件 (labels.tsv)

用法:
    python3 scripts/generate_labels.py --ms-dir data/ZGC/test --output data/ZGC/test/labels.tsv
"""
import argparse
from pathlib import Path


def parse_ms_file(ms_path: Path) -> dict:
    """解析单个 .ms 文件的元数据头"""
    meta = {}
    with open(ms_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">compound"):
                meta["compound"] = line[len(">compound"):].strip()
            elif line.startswith(">formula"):
                meta["formula"] = line[len(">formula"):].strip()
            elif line.startswith(">parentmass"):
                meta["parentmass"] = line[len(">parentmass"):].strip()
            elif line.startswith(">ionization"):
                meta["ionization"] = line[len(">ionization"):].strip()
            elif line.startswith("#smiles") or line.startswith(">smiles"):
                meta["smiles"] = line.split(maxsplit=1)[1].strip() if " " in line else ""
            elif line.startswith(">InChIKey"):
                meta["inchikey"] = line[len(">InChIKey"):].strip()
            elif line.startswith(">ms2peaks"):
                # ms2peaks 标志着头信息的结束
                break
    return meta


def main():
    parser = argparse.ArgumentParser(description="从 .ms 文件生成 MIST 标签文件")
    parser.add_argument("--ms-dir", required=True, help="包含 .ms 文件的目录")
    parser.add_argument("--output", required=True, help="输出 labels.tsv 路径")
    parser.add_argument("--dataset", default="ZGC", help="数据集名称（默认: ZGC）")
    parser.add_argument("--instrument", default="unknown", help="仪器类型（默认: unknown）")
    args = parser.parse_args()

    ms_dir = Path(args.ms_dir)
    if not ms_dir.is_dir():
        print(f"错误: 目录不存在 {ms_dir}")
        return

    ms_files = sorted(ms_dir.glob("*.ms"))
    if not ms_files:
        print(f"错误: 在 {ms_dir} 中未找到 .ms 文件")
        return

    print(f"找到 {len(ms_files)} 个 .ms 文件")

    # 列定义（MIST 期望的格式）
    columns = [
        "spec",        # 谱图名称（不含 .ms 后缀）
        "formula",     # 分子式
        "ionization",  # 电离方式，如 [M+H]+
        "dataset",     # 数据集名称
        "compound",    # 化合物名称
        "parentmass",  # 母离子质量
        "instrument",  # 仪器类型
        "smiles",      # SMILES（可选）
        "inchikey",    # InChIKey（可选）
    ]

    # 收集所有条目
    rows = []
    missing = {"formula": 0, "ionization": 0}
    for ms_path in ms_files:
        spec_name = ms_path.stem  # 文件名去掉 .ms 后缀
        meta = parse_ms_file(ms_path)

        if "formula" not in meta:
            print(f"  ⚠ 跳过 {ms_path.name}: 未找到 >formula")
            missing["formula"] += 1
            continue
        if "ionization" not in meta:
            print(f"  ⚠ 跳过 {ms_path.name}: 未找到 >ionization")
            missing["ionization"] += 1
            continue

        row = {
            "spec": spec_name,
            "formula": meta.get("formula", ""),
            "ionization": meta.get("ionization", ""),
            "dataset": args.dataset,
            "compound": meta.get("compound", spec_name),
            "parentmass": meta.get("parentmass", "0"),
            "instrument": args.instrument,
            "smiles": meta.get("smiles", ""),
            "inchikey": meta.get("inchikey", ""),
        }
        rows.append(row)

    # 写入 TSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # 表头
        f.write("\t".join(columns) + "\n")
        # 数据行
        for row in rows:
            f.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")

    total = len(rows)
    print(f"\n✅ 已生成标签文件: {output_path.resolve()}")
    print(f"   共 {total} 条谱图")
    if missing["formula"]:
        print(f"   ⚠ {missing['formula']} 个文件缺少 >formula，已跳过")
    if missing["ionization"]:
        print(f"   ⚠ {missing['ionization']} 个文件缺少 >ionization，已跳过")

    # 打印前 3 行预览
    print("\n📋 预览（前 3 行）:")
    with open(output_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 4:  # 1 表头 + 3 数据行
                break
            print(f"   {line.rstrip()}")


if __name__ == "__main__":
    main()
