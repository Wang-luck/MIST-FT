""" train_mist_my.py

只微调自注意力层，其余全部冻结。
适合目标数据与预训练数据碎裂模式不同（如中药vs代谢物）的场景。

用法:
    python3 src/mist/train_mist_my.py \
        --spec-folder data/ZGC/train \
        --labels-file data/ZGC/train_labels.tsv
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
    """run_training."""
    # Get args
    args = get_args()
    kwargs = args.__dict__
    save_dir = Path(kwargs.get("save_dir"))
    utils.setup_train(save_dir, kwargs)

    # Split data
    my_splitter = splitter.get_splitter(**kwargs)

    # Get model class
    model_class = mist_model.MistNet
    kwargs["model"] = model_class.__name__
    kwargs["spec_features"] = model_class.spec_features()
    kwargs["mol_features"] = model_class.mol_features()
    kwargs["dataset_type"] = model_class.dataset_type()

    # Get featurizers
    paired_featurizer = featurizers.get_paired_featurizer(**kwargs)

    # Build dataset
    spectra_mol_pairs = datasets.get_paired_spectra(**kwargs)
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Redefine splitter s.t. this splits three times and remove subsetting
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

    # Create model
    model = model_class(**kwargs)

    if kwargs.get("ckpt_file") is not None:
        model.load_from_ckpt(**kwargs)

    # 只训练自注意力层，其余全部冻结
    for name, param in model.named_parameters():
        if "peak_attn_layers" in name and param.is_floating_point():
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logging.info(f"Trainable params: {trainable:,} ({trainable/(trainable+frozen)*100:.1f}%)")
    logging.info(f"Frozen params: {frozen:,}")

    logging.info(f"Starting fold: {split_name}")

    test_loss = model.train_model(
        spec_dataloader_module,
        log_name="",
        log_version=split_name,
        **kwargs,
    )

    # for each dict, add split name
    for j in test_loss:
        j.update({"split_name": split_name})

    # Export train dataset names
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
