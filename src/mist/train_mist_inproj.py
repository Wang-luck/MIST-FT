""" train_mist_inproj.py

只微调自注意力层的 in_proj_weight（QKV 投影），其余全冻结。
参数量：393,216（2.6%）

作用：调整注意力头的"关注点"——每个头去注意什么碎片关系。
"""
import yaml
import logging
import pickle
from pathlib import Path
import argparse

from mist.models import mist_model
from mist.data import datasets, splitter, featurizers
from mist import utils, parsing


def get_args():
    parser = argparse.ArgumentParser(add_help=True)
    parsing.add_base_args(parser)
    parsing.add_dataset_args(parser)
    parsing.add_train_args(parser)
    parsing.add_mist_args(parser)
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

    # 只训练 in_proj_weight，其余全部冻结
    for name, param in model.named_parameters():
        if "in_proj_weight" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable (in_proj_weight only): {trainable:,} ({trainable/total*100:.2f}%)")

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
