import argparse
import logging
import sys
import numpy as np

import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline

from src.precontext.utils import generate_with_diffuser

LOG = logging.getLogger(__name__)


def setup_args():
    parser = argparse.ArgumentParser(description="Create multimodal sst")
    parser.add_argument("--hf_repository",
                        type=str,
                        default="israfelsr/multimodal_sst",
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
    if args.percentage:
        sst = load_dataset("sst2",
                           "default",
                           split=f"{args.split}[:{args.percentage}%]")
    else:
        sst = load_dataset("sst2", "default", split=args.split)
    multimodal_sst = sst.map(lambda sample: generate_with_diffuser(
        sample, diffuser, latents, generator, args))
    LOG.info(
        f"Dataset created correctly with features {multimodal_sst.features}")

    LOG.info(
        f"Pushing dataset to {args.hf_repository} hugging-face repository")
    multimodal_sst.push_to_hub("israfelsr/multimodal_sst",
                               private=True,
                               use_auth_token=args.auth_token,
                               split=args.split)


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