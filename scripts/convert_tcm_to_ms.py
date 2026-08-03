#!/usr/bin/env python3
"""
将 TCM-MS2Link 的 TSV 转换为 .ms 文件 + labels.tsv

输入格式: identifier, mzs(逗号分隔), intensities(逗号分隔),
          smiles, inchikey, formula, parent_mass, precursor_mz, adduct, instrument

用法:
    python3 scripts/convert_tcm_to_ms.py \
        --tsv data/TCM-MS2Link/TCM-MS2Clean.tsv \
        --output-dir data/tcm_ms/
"""
import argparse
import csv
import re
from pathlib import Path

# MIST 支持的元素
VALID_ELEMENTS = {
    "C", "H", "As", "B", "Br", "Cl", "Co", "F", "Fe",
    "I", "K", "N", "Na", "O", "P", "S", "Se", "Si",
}

# adduct 映射
ADDUCT_MAP = {
    "[M+H]+": "[M+H]+",
    "[M+Na]+": "[M+Na]+",
    "[M+K]+": "[M+K]+",
    "[M-H2O+H]+": "[M-H2O+H]+",
    "[M-2H2O+H]+": "[M-H4O2+H]+",
    "[M+NH4]+": "[M+H3N+H]+",
}


def has_unsupported_elements(formula: str) -> bool:
    """检查化学式中是否有 MIST 不支持的金属元素"""
    elems = re.findall(r"[A-Z][a-z]*", formula)
    return any(e not in VALID_ELEMENTS for e in elems)


def main():
    parser = argparse.ArgumentParser(description="TCM-MS2Link TSV → .ms + labels")
    parser.add_argument("--tsv", required=True, help="输入 TSV 文件")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--max-spectra", type=int, default=None, help="最多处理谱图数")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ms_dir = out_dir / "ms"
    ms_dir.mkdir(parents=True, exist_ok=True)

    # 读取 TSV
    rows = []
    with open(args.tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if args.max_spectra and i >= args.max_spectra:
                break
            rows.append(row)

    filtered_adduct = 0
    filtered_elem = 0
    filtered_peak = 0
    labels = []

    for row in rows:
        spec_id = row["identifier"]
        smiles = row["smiles"]
        formula = row["formula"]
        adduct = row["adduct"].strip()
        parent_mass = row["parent_mass"]
        instrument = row.get("instrument", "unknown")
        inchikey = row.get("inchikey", "")
        mzs_str = row.get("mzs", "")
        inten_str = row.get("intensities", "")

        # ① 过滤不支持的 adduct
        adduct = ADDUCT_MAP.get(adduct)
        if adduct is None:
            filtered_adduct += 1
            continue

        # ② 过滤含不支持元素的化学式（如 Mg、Al 等）
        if has_unsupported_elements(formula):
            filtered_elem += 1
            continue

        # ③ 解析峰
        mz_list = [float(x) for x in mzs_str.split(",") if x.strip()]
        inten_list = [float(x) for x in inten_str.split(",") if x.strip()]

        # 跳过无有效峰的数据
        if len(mz_list) == 0 or len(inten_list) == 0:
            filtered_peak += 1
            continue

        # 归一化强度
        if inten_list and max(inten_list) > 0:
            max_inten = max(inten_list)
            inten_list = [i / max_inten for i in inten_list]

        # 写入 .ms 文件
        ms_path = ms_dir / f"{spec_id}.ms"
        with open(ms_path, "w") as f:
            f.write(f">compound {spec_id}\n")
            f.write(f">formula {formula}\n")
            f.write(f">parentmass {parent_mass}\n")
            f.write(f">ionization {adduct}\n")
            f.write(f">smiles {smiles}\n")
            if inchikey:
                f.write(f">InChIKey {inchikey}\n")
            f.write(f">instrument {instrument}\n")
            f.write(f"\n>ms2peaks\n")
            for mz, inten in zip(mz_list, inten_list):
                f.write(f"{mz:.4f} {inten:.6f}\n")

        labels.append({
            "spec": spec_id, "formula": formula, "ionization": adduct,
            "dataset": "TCM-MS2Link", "compound": spec_id,
            "parentmass": parent_mass, "instrument": instrument,
            "smiles": smiles, "inchikey": inchikey,
        })

    # 写入 labels.tsv
    label_path = out_dir / "labels.tsv"
    cols = ["spec", "formula", "ionization", "dataset", "compound",
            "parentmass", "instrument", "smiles", "inchikey"]
    with open(label_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in labels:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    print(f"总谱图: {len(labels)}")
    print(f"  .ms 文件: {ms_dir}/")
    print(f"  标签文件: {label_path}")
    print(f"过滤:")
    print(f"  不支持 adduct: {filtered_adduct}")
    print(f"  不支持元素:    {filtered_elem}")
    print(f"  无有效峰:      {filtered_peak}")


if __name__ == "__main__":
    main()
