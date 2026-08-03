#!/usr/bin/env python3
"""
将 LoRA 训练产生的 checkpoint 合并为干净的 MistNet 权重，
使 pred_fp.py 可以直接加载推理。

用法:
    python3 scripts/merge_lora_checkpoint.py \
        --lora-ckpt results/ZGC/mist_ft_lora/split/best.ckpt \
        --output results/ZGC/mist_ft_lora/split/best_merged.ckpt
"""
import argparse
import re
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA checkpoint 为干净权重")
    parser.add_argument("--lora-ckpt", required=True, help="LoRA 训练产生的 best.ckpt")
    parser.add_argument("--output", required=True, help="输出合并后的 checkpoint 路径")
    args = parser.parse_args()

    ckpt = torch.load(args.lora_ckpt, map_location="cpu")
    state = ckpt["state_dict"]

    # 收集需要合并的信息
    linear_orig = {}   # key → (weight_key, clean_key)
    lora_As = {}       # key → lora_A_key
    lora_Bs = {}       # key → lora_B_key

    for key in state.keys():
        if key.endswith(".linear.weight"):
            # key 如: ...linear1.linear.weight 或 ...out_proj.linear.weight
            # clean key: ...linear1.weight 或 ...out_proj.weight
            base = key.rsplit(".linear.weight", 1)[0]  # ...linear1 或 ...out_proj
            # 替换 .linear. 为空 → ...linear1 不变，...out_proj 不变
            # 但 linear1 不含 .linear. （它包含 .linear1）
            clean_w = base + ".weight"
            clean_b = base + ".bias"
            bias_key = base + ".linear.bias"
            linear_orig[base] = {
                "w_key": key, "b_key": bias_key,
                "clean_w": clean_w, "clean_b": clean_b,
            }
        elif key.endswith("lora_A_in_proj"):
            base = key.rsplit("lora_A_in_proj", 1)[0]
            lora_As[base + "in_proj"] = key
        elif key.endswith("lora_B_in_proj"):
            base = key.rsplit("lora_B_in_proj", 1)[0]
            lora_Bs[base + "in_proj"] = key
        elif key.endswith(".lora_A"):
            base = key.rsplit(".lora_A", 1)[0]
            lora_As[base] = key
        elif key.endswith(".lora_B"):
            base = key.rsplit(".lora_B", 1)[0]
            lora_Bs[base] = key

    # 构建合并后的 state_dict
    merged = {}
    for key, tensor in state.items():
        if "lora_" in key:
            continue
        if key.endswith(".linear.weight"):
            clean = key.rsplit(".linear.weight", 1)[0] + ".weight"
            merged[clean] = tensor.clone()
        elif key.endswith(".linear.bias"):
            clean = key.rsplit(".linear.bias", 1)[0] + ".bias"
            merged[clean] = tensor.clone()
        else:
            merged[key] = tensor.clone()

    # 合并 LinearLoRA 包装器的权重: W_final = W_orig + B@A
    for base, info in linear_orig.items():
        if base in lora_As and base in lora_Bs:
            merged[info["clean_w"]] += state[lora_Bs[base]] @ state[lora_As[base]]
            merged[info["clean_w"]] = merged[info["clean_w"]].contiguous()
            print(f"  ✅ 合并 LinearLoRA {base}")

    # 合并 in_proj_weight 的 LoRA
    for base in list(lora_As.keys()):
        if base.endswith("in_proj") and base in lora_Bs:
            # base = ...self_attn
            in_proj_key = base + "in_proj_weight"
            if in_proj_key in merged:
                merged[in_proj_key] += state[lora_Bs[base]] @ state[lora_As[base]]
                merged[in_proj_key] = merged[in_proj_key].contiguous()
                print(f"  ✅ 合并 in_proj_weight {base}")

    # 保存（保留原 checkpoint 的 hyper_parameters）
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_ckpt = {"state_dict": merged, "epoch": "merged"}
    if "hyper_parameters" in ckpt:
        new_ckpt["hyper_parameters"] = ckpt["hyper_parameters"]
    torch.save(new_ckpt, out_path)

    # 检查 key 是否合理
    wrapper_keys = [k for k in merged if ".linear." in k]
    if wrapper_keys:
        print(f"\n⚠  发现 {len(wrapper_keys)} 个包装器 key 未清理")
        for k in wrapper_keys[:5]:
            print(f"   {k}")

    print(f"\n✅ 合并完成，输出: {out_path}")
    print(f"   state_dict keys: {len(merged)}")


if __name__ == "__main__":
    main()
