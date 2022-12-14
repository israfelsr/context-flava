import logging
import sys
import numpy as np

import torch
from datasets import load_dataset, concatenate_datasets

from src.precontext.utils import generate_with_diffuser
from src.utils import build_general_config

LOG = logging.getLogger(__name__)


def main():
    config = build_general_config()
    #sst = load_dataset("sst", split="train")
    diffuser = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", use_auth_token=config.auth_token)
    diffuser.to(config.device)

    generator = torch.Generator(device=config.device)
    generator.manual_seed(config.seed)

    # Fixed latents for the image generation
    latents = torch.randn((1, diffuser.unet.in_channels, 64, 64),
                          generator=generator,
                          device=config.device)
    chunks = np.linspace(0, 100, config.num_chunks + 1)
    dataset_stats = {
        "split":
        config.split,
        "num_chunks":
        config.num_chunks,
        "chunks":
        chunks,
        "num_chunks_processed":
        config.num_chunks_processed,
        "splits_in_repo":
        [f"{config.split}_{t}" for t in range(config.num_chunks_processed)],
    }

    for t, (start, stop) in enumerate(zip(chunks[:-1], chunks[1:])):
        # Check if the chunk was already processed
        if t < config.num_chunks_processed:
            continue
        LOG.info(f"Loading chunk from {start}% to {stop}%")
        image_net = load_dataset("Maysee/tiny-imagenet",
                                 split=f"train[{int(start)}%:{int(stop)}%]")

        multimodal_sst2 = sst2.map(lambda batch: generate_with_diffuser(
            batch, diffuser, latents, generator, config),
                                   batched=True,
                                   batch_size=config.batch_size)

        LOG.info(f"Dataset chunk processing finished")
        LOG.info(f"Pushing chunk to HF respository: {config.repo_id}")
        repo_split = f"{config.split}_{t}"
        LOG.info(f"Storing the chunk using the split: {repo_split}")
        multimodal_sst2.push_to_hub(
            repo_id=config.repo_id,
            split=repo_split,
            private=True,
            token=config.auth_token,
        )
        dataset_stats["num_chunks_processed"] += 1
        LOG.info(
            f"{dataset_stats['num_chunks_processed']}/{config.num_chunks} Chunks processed"
        )
        dataset_stats["splits_in_repo"].append(repo_split)
        LOG.info(dataset_stats)

    if config.do_merge:
        assert dataset_stats["num_chunks_processed"] == config.num_chunks
        LOG.info(f"All chunks processed, merging into split: {config.split}")
        prev_data = load_dataset(config.repo_id,
                                 split=dataset_stats["splits_in_repo"][0],
                                 use_auth_token=config.auth_token)
        for next_split in dataset_stats["splits_in_repo"][1:]:
            next_data = load_dataset(config.repo_id,
                                     split=next_split,
                                     use_auth_token=config.auth_token)
            prev_data = concatenate_datasets([prev_data, next_data])
        LOG.info("Pushing all merged dataset to hub")
        prev_data.push_to_hub(repo_id=config.repo_id,
                              split=config.split,
                              private=True,
                              token=config.auth_token)


if __name__ == "__main__":
    # Set up logger
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    LOG.setLevel(log_level)

    main()