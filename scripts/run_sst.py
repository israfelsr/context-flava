import argparse
from genericpath import isdir
import logging
import os
import sys

import datasets
import transformers

LOG = logging.getLogger(__name__)


def setup_args():
    parser = argparse.ArgumentParser(description="Finetuning on SST")
    parser.add_argument("--data_root", type=str, help="Path to data folder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_train", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = setup_args()

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

    # Detecting the last checkpoint
    last_checkpoint = None

    # Initialize our datasets
