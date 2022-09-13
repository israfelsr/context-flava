import argparse
from statistics import multimode

import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline

from src.precontext.utils import generate_with_diffuser


def main(args: argparse.Namespace):
    sst = load_dataset("sst", split="train")
    diffuser = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", use_auth_token=args.auth_token)
    diffuser.to(args.device)

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)

    latents = torch.randn((1, diffuser.unet.in_channels, 64, 64),
                          generator=generator,
                          device=args.device)

    multimodal_sst = sst.map(lambda sample: generate_with_diffuser(
        sample, diffuser, latents, generator, args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth_token",
                        type=str,
                        help="HF authorization token for writing and reading")
    parser.add_argument("--device",
                        type=str,
                        help="Device to use: cuda or cpu")
    parser.add_argument("--seed",
                        type=int,
                        help="Set seed for reproducibility")
    parser.add_argumnet("--max_nsfw_tries",
                        type=int,
                        help="Max retries after nsfw warning")
    parser.add_argument()
    args = parser.parse_args()
    main(args)
