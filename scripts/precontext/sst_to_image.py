import argparse
import logging
import sys
import numpy as np

import torch
from datasets import load_dataset, concatenate_datasets
from diffusers import StableDiffusionPipeline

from src.precontext.utils import generate_with_diffuser

LOG = logging.getLogger(__name__)


def setup_args():
    parser = argparse.ArgumentParser(description="Create multimodal sst")
    parser.add_argument("--hf_repository",
                        type=str,
                        default="israfelsr/multimodal_sst2",
                        help="HuggingFace repository")
    parser.add_argument("--auth_token",
                        type=str,
                        required=True,
                        help="HF authorization token for writing and reading")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="Device to use: cuda or cpu")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Set seed for reproducibility")
    parser.add_argument("--max_nsfw_tries",
                        type=int,
                        default=5,
                        help="Max tries to avoid nsfw warning")
    parser.add_argument("--split",
                        type=str,
                        required=True,
                        help="Split of the dataset to process")
    parser.add_argument("--percentage",
                        type=int,
                        help="Percentage of dataset if testing")
    parser.add_argument("--num_chunks",
                        type=int,
                        default=None,
                        help="Number of divisions to create the dataset")
    args = parser.parse_args()
    return args


def main():
    args = setup_args()
    #sst = load_dataset("sst", split="train")
    diffuser = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", use_auth_token=args.auth_token)
    diffuser.to(args.device)

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)

    # Fixed latents for the image generation
    latents = torch.randn((1, diffuser.unet.in_channels, 64, 64),
                          generator=generator,
                          device=args.device)
    chunks = np.linspace(0, 100, args.num_chunks)
    dataset_stats = {
        "split": args.split,
        "num_chunks": args.num_chunks,
        "chunks": chunks,
        "num_chunk_processed": 0,
        "splits_in_repo": []
    }

    for t, (start, stop) in enumerate(zip(chunks[:-1], chunks[1:])):
        LOG.info(f"Loading chunk from {start}% to {stop}%")
        sst2 = load_dataset("sst2",
                            "default",
                            split=f"{args.split}[{int(start)}%:{int(stop)}%]")
        multimodal_sst2 = sst2.map(lambda sample: generate_with_diffuser(
            sample, diffuser, latents, generator, args))
        LOG.info(f"Dataset finished with features: {multimodal_sst2.features}")
        LOG.info(f"Pushing dataset to HF respository: {args.hf_repository}")
        repo_split = f"{args.split}_{t}"
        LOG.info(f"Storing the dataset in split {repo_split}")
        multimodal_sst2.push_to_hub(
            repo_id=args.hf_repository,
            split=repo_split,
            private=True,
            token=args.auth_token,
        )
        dataset_stats["num_chunks_processed"] = t
        dataset_stats["splits_in_repo"].append(repo_split)
        LOG.info(dataset_stats)
    LOG.info(f"All chunks processed merging into split: {args.split}")
    prev_data = load_dataset(repo_id=args.hf_repository,
                             split=dataset_stats["splits_in_repo"][0],
                             use_auth_token=args.auth_token)
    for next_split in dataset_stats["splits_in_repo"][1:]:
        next_data = load_dataset(repo_id=args.hf_repository,
                                 split=next_split,
                                 use_auth_token=args.auth_token)
        prev_data = concatenate_datasets([prev_data, next_data])
    LOG.info("Pushing all merged dataset to hub")
    prev_data.push_to_hub(repo_id=args.hf_repository,
                          split=args.split,
                          private=True,
                          token=args.auth_token)
    '''
    if args.percentage:
        sst2 = load_dataset("sst2",
                            "default",
                            split=f"{args.split}[:{args.percentage}%]")
    else:
        sst2 = load_dataset("sst2", "default", split=args.split)
    multimodal_sst2 = sst2.map(lambda sample: generate_with_diffuser(
        sample, diffuser, latents, generator, args))
    LOG.info(
        f"Dataset created correctly with features {multimodal_sst2.features}")

    LOG.info(
        f"Pushing dataset to {args.hf_repository} hugging-face repository")
    multimodal_sst2.push_to_hub("israfelsr/multimodal_sst2",
                                private=True,
                                token=args.auth_token,
                                split=args.split)
    '''


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