# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List

import requests
from datasets import concatenate_datasets, load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from flava.definitions import HFDatasetInfo
from PIL import Image, UnidentifiedImageError


DATASETS_USER_AGENT = get_datasets_user_agent()


def build_datasets_from_info(dataset_infos: List[HFDatasetInfo], split: str = "train"):
    dataset_list = []
    for dataset_info in dataset_infos:
        current_dataset = load_dataset(
            dataset_info.key,
            dataset_info.subset,
            split=dataset_info.split_key_mapping[split],
            use_auth_token=True,
            **dataset_info.extra_kwargs,
        )
        if dataset_info.remove_columns is not None:
            current_dataset = current_dataset.remove_columns(
                dataset_info.remove_columns
            )
        if dataset_info.rename_columns is not None:
            for rename in dataset_info.rename_columns:
                current_dataset = current_dataset.rename_column(rename[0], rename[1])

        dataset_list.append(current_dataset)

    return concatenate_datasets(dataset_list)


def fetch_single_image(image_url, timeout, retries=0, sleep_timer=0):
    for _ in range(retries + 1):
        try:
            image = Image.open(
                requests.get(
                    image_url,
                    stream=True,
                    headers={"user-agent": DATASETS_USER_AGENT},
                    timeout=timeout,
                ).raw
            )
            break
        except (requests.exceptions.ConnectionError, UnidentifiedImageError):
            image = None
            time.sleep(sleep_timer)

    return image


def fetch_images(batch, num_threads, timeout=None, retries=0, sleep_timer=0):
    if "image" in batch:
        # This dataset already has "image" defined.
        return batch
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(
            executor.map(
                partial(
                    fetch_single_image,
                    timeout=timeout,
                    retries=retries,
                    sleep_timer=sleep_timer,
                ),
                batch["image_url"],
            )
        )
    return batch

def add_black_images(batch):
    if "image" in batch:
        # This dataset already has "image" defined.
        return batch
    batch["image"] = [Image.new('RGB', (224, 224))] * len(batch['text'])
    return batch

def add_random_image(batch, percentage=0.5):
    if "image" in batch:
        assert percentage is not None
        if percentage > np.random.rand(1):
            noisy_images = np.random.randint(
                    low=0, 
                    high=256,
                    size=(len(batch['text']), 224, 224, 3),
                    dtype=np.uint8)
            batch["image"] = [Image.fromarray(noisy_images[i]) for i in range(len(batch['text']))]
    return batch


def add_empty_text(batch):
    if "text" in batch:
        return batch
    batch["text"] = [""] * len(batch["image"])
    return batch
