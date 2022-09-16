import argparse
import logging
import os
import sys

import datasets
import transformers

from flava.data import TorchVisionDataModule, TextDataModule
from flava.data.datamodules import VLDataModule
from flava.definitions import FLAVAArguments
from flava.utils import build_config, build_datamodule_kwargs

LOG = logging.getLogger(__name__)


def main():
    config: FLAVAArguments = build_config()
    # TODO: check the arguments in FLAVA
    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    LOG.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Setting up the datamodule
    # TODO: check the arguments for the datasets, example:
    ''' {'train': [{'key': 'israfelsr/multimodal_sst',
                    'remove_columns': None,
                    'rename_columns': ["tokens", "tree"], 
                    'split_key_mapping': {'train': 'train',
                                          'validation': 'validation',
                                          'test': 'test'}, 'extra_kwargs': {}}], 
         'val': None, 'test': None, 'batch_size': None, 'num_workers': None,
         'allow_uneven_batches': False,
         'datamodule_extra_kwargs': {'text_columns': ['sentence']}}'''
    assert len(config.datasets.selected) == 1
    # TODO: when trianing multimodal/unimodal change config.dataset.selected
    if "image" in config.datasets.selected:
        datamodule = TorchVisionDataModule(
            **build_datamodule_kwargs(config.datasets.image, config.training))
    elif "text":
        datamodule = TextDataModule(
            **build_datamodule_kwargs(config.datasets.text, config.training))
    else:
        datamodule = VLDataModule(
            **build_datamodule_kwargs(config.datasets.vl, config.training),
            finetuning=True,
        )

    datamodule.setup("fit")

    # Testing dataset
    LOG.info("Testing dataset")
    LOG.info(f"Dataset {datamodule}")
    #for sample in datamodule:
    #    LOG.info(f"One sample: {sample}")
    #    break

    # Detecting the last checkpoint
    last_checkpoint = None

    # Loading the datamodules


if __name__ == "__main__":
    main()
