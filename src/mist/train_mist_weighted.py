""" train_mist_weighted.py

频率加权微调：对指纹低频位给予更高损失权重。
"""
import yaml
import logging
import pickle
import numpy as np
from pathlib import Path
import argparse
import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):
    """带逐位权重的 BCE 损失"""
    def __init__(self, base_loss, weights):
        super().__init__()
        self.base_loss = base_loss
        self.register_buffer("bit_weights", weights)

    def forward(self, pred, target):
        loss = self.base_loss(pred, target)
        # 只在维度匹配时加权（4096 位），中间迭代（256/512/1024/2048）不加权
        if loss.shape[-1] == self.bit_weights.shape[0]:
            loss = loss * self.bit_weights.to(loss.device)
        return loss

from mist.models import mist_model
from mist.data import datasets, splitter, featurizers
from mist import utils, parsing


def get_args():
    parser = argparse.ArgumentParser(add_help=True)
    parsing.add_base_args(parser)
    parsing.add_dataset_args(parser)
    parsing.add_train_args(parser)
    parsing.add_mist_args(parser)
    parser.add_argument("--bit-weights", default=None,
                        help="每位权重 .npy 文件（由 compute_bit_weights.py 生成）")
    return parser.parse_args()


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

    # 加载位权重并注入模型
    if kwargs.get("bit_weights"):
        weights = torch.tensor(np.load(kwargs["bit_weights"]), dtype=torch.float32)
        model.loss_fn = WeightedBCELoss(model.bce_loss, weights)
        logging.info(f"Loaded bit weights: {weights.shape}, "
                     f"range [{weights.min():.2f}, {weights.max():.2f}]")

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
