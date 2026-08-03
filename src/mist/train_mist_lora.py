""" train_mist_lora.py

LoRA 微调：用包装器替换注意力层的 Linear 模块，
添加低秩适配矩阵 B@A，使梯度可正常回传。
其余参数全部冻结。适合小数据量微调。

用法:
    python3 src/mist/train_mist_lora.py \
        --spec-folder data/ZGC/train \
        --labels-file data/ZGC/train_labels.tsv \
        --lora-rank 8
"""
import yaml
import logging
import pickle
import math
from pathlib import Path
import argparse

import torch
import torch.nn as nn

from mist.models import mist_model
from mist.data import datasets, splitter, featurizers
from mist import utils, parsing


def get_args():
    parser = argparse.ArgumentParser(add_help=True)
    parsing.add_base_args(parser)
    parsing.add_dataset_args(parser)
    parsing.add_train_args(parser)
    parsing.add_mist_args(parser)
    parser.add_argument(
        "--lora-rank", type=int, default=8,
        help="LoRA rank (默认 8)",
    )
    return parser.parse_args()


class LinearLoRA(nn.Module):
    """包装 nn.Linear，添加 LoRA 低秩适配"""

    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

        out_dim, in_dim = linear.weight.shape
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T)


def _get_module(model: nn.Module, name: str) -> nn.Module:
    for part in name.split("."):
        model = model[int(part)] if part.isdigit() else getattr(model, part)
    return model


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    """按路径设置子模块"""
    parts = name.split(".")
    parent = _get_module(model, ".".join(parts[:-1]))
    key = int(parts[-1]) if parts[-1].isdigit() else parts[-1]
    if isinstance(parent, nn.ModuleList) or isinstance(parent, nn.Sequential):
        parent[key] = new_module
    else:
        setattr(parent, key, new_module)


def inject_lora(model: nn.Module, rank: int):
    """用 LinearLoRA 包装器替换注意力层输出投影和 FFN 中的 Linear 模块"""
    # 要替换的模块（参数名中的模块路径末尾匹配）
    replace_suffixes = [
        "self_attn.out_proj",   # 注意力输出投影 [256×256]
        "linear1",               # FFN 第一层 [1024×256]
        "linear2",               # FFN 第二层 [256×1024]
    ]

    lora_param_count = 0
    replaced = []

    for param_name, _ in list(model.named_parameters()):
        if "peak_attn_layers" not in param_name:
            continue
        if not param_name.endswith(".weight"):
            continue

        matched = [s for s in replace_suffixes if param_name.endswith(s + ".weight")]
        if not matched:
            continue

        module_path = ".".join(param_name.split(".")[:-1])  # 去掉最后的 "weight"
        linear = _get_module(model, module_path)
        if not isinstance(linear, nn.Linear):
            continue

        lora_linear = LinearLoRA(linear, rank)
        _set_module(model, module_path, lora_linear)

        params_here = lora_linear.lora_A.numel() + lora_linear.lora_B.numel()
        lora_param_count += params_here
        replaced.append((module_path, linear.weight.shape, params_here))
        logging.info(f"  LoRA → {module_path:<55} "
                     f"[{linear.weight.shape[0]:>4}×{linear.weight.shape[1]:<4}]  {params_here:,}")

    # 处理 in_proj_weight: 用 forward_pre_hook 在每次前向时临时替换
    for name, module in model.named_modules():
        if "peak_attn_layers" not in name or not name.endswith("self_attn"):
            continue
        if not hasattr(module, "in_proj_weight"):
            continue

        out_dim, in_dim = module.in_proj_weight.shape
        module.in_proj_weight.requires_grad = False
        orig_data = module.in_proj_weight.data.clone()

        lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))

        module.register_parameter("lora_A_in_proj", lora_A)
        module.register_parameter("lora_B_in_proj", lora_B)
        lora_param_count += lora_A.numel() + lora_B.numel()

        # 前向 hook：把 in_proj_weight 替换为 LoRA 增强版，后向恢复
        def pre_hook(a, b, orig):
            def hook(module, input):
                module.in_proj_weight = nn.Parameter(orig + (b @ a))
            return hook

        def post_hook(module, input, output):
            module.in_proj_weight = nn.Parameter(orig_data)

        module.register_forward_pre_hook(pre_hook(lora_A, lora_B, orig_data))
        module.register_forward_hook(post_hook)

        logging.info(f"  LoRA → {name}.in_proj_weight                    "
                     f"[{out_dim:>4}×{in_dim:<4}]  {lora_A.numel()+lora_B.numel():,}")

    # 冻结全部 → 仅解冻 LoRA 参数
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name

    return lora_param_count


def run_training():
    args = get_args()
    kwargs = args.__dict__
    save_dir = Path(kwargs.get("save_dir"))
    utils.setup_train(save_dir, kwargs)

    my_splitter = splitter.get_splitter(**kwargs)
    model_class = mist_model.MistNet
    kwargs["model"] = model_class.__name__
    kwargs["spec_features"] = model_class.spec_features()
    kwargs["mol_features"] = model_class.mol_features()
    kwargs["dataset_type"] = model_class.dataset_type()

    paired_featurizer = featurizers.get_paired_featurizer(**kwargs)
    spectra_mol_pairs = datasets.get_paired_spectra(**kwargs)
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))
    split_name, (train, val, test) = my_splitter.get_splits(spectra_mol_pairs)
    for name, _data in zip(["train", "val", "test"], [train, val, test]):
        logging.info(f"Len of {name}: {len(_data)}")

    train_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=train, featurizer=paired_featurizer, **kwargs
    )
    val_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=val, featurizer=paired_featurizer, **kwargs
    )
    test_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=test, featurizer=paired_featurizer, **kwargs
    )
    spec_dataloader_module = datasets.SpecDataModule(
        train_dataset, val_dataset, test_dataset, **kwargs
    )

    model = model_class(**kwargs)
    if kwargs.get("ckpt_file") is not None:
        model.load_from_ckpt(**kwargs)

    rank = kwargs.get("lora_rank", 8)
    logging.info(f"Injecting LoRA (rank={rank}) into attention layers...")
    lora_total = inject_lora(model, rank)

    total_all = sum(p.numel() for p in model.parameters())
    logging.info(f"Total params: {total_all:,}")
    logging.info(f"LoRA params (trainable): {lora_total:,} ({lora_total/total_all*100:.3f}%)")

    logging.info(f"Starting fold: {split_name}")
    test_loss = model.train_model(
        spec_dataloader_module, log_name="", log_version=split_name, **kwargs,
    )
    for j in test_loss:
        j.update({"split_name": split_name})

    all_train_spec_names = [
        *train_dataset.get_spectra_names(),
        *val_dataset.get_spectra_names(),
    ]
    with open(Path(model.results_dir) / "train_spec_names.p", "wb") as fp:
        pickle.dump(all_train_spec_names, fp)

    output_dict = {"args": kwargs, "results": test_loss}
    output_str = yaml.dump(output_dict, indent=2, default_flow_style=False)
    with open(save_dir / "results.yaml", "w") as fp:
        fp.write(output_str)


if __name__ == "__main__":
    import time
    start_time = time.time()
    run_training()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
