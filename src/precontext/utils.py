import argparse
import logging
import torch

#from datasets import load_dataset, concatenate_datasets
from diffusers import StableDiffusionPipeline

LOG = logging.getLogger(__name__)


def generate_with_diffuser(sample, diffuser: StableDiffusionPipeline,
                           latents: torch.tensor, generator: torch.Generator,
                           args: argparse.Namespace):
    with torch.autocast(args.device):
        image = diffuser(
            [sample["sentence"]],
            guidance_scale=7.5,
            latents=latents,
        )
    idx = 0
    while image["nsfw_content_detected"][0]:
        generator = generator.manual_seed(generator.seed() + 1)
        latents = torch.randn((1, diffuser.unet.in_channels, 64, 64),
                              generator=generator,
                              device="cuda")
        with torch.autocast("cuda"):
            idx = +1
            image = diffuser(
                [sample["sentence"]] * 1,
                guidance_scale=7.5,
                latents=latents,
            )
        if idx == args.max_nsfw_tries:
            LOG.info("NSFW detected and max number of tries reached")
            LOG.info("Image set to black")
            break
    generator.manual_seed(args.seed)
    sample["image"] = image["sample"][0]
    return sample


def push_chunk_to_hub(chunk_dataset, split, dataset_stats, args):
    # Load the current state of the dataset
    LOG.info(f"Storing the dataset in split {split}")

    #new_dataset = concatenate_datasets(
    #    [old_dataset[args.split], chunk_dataset[args.split]])
    LOG.info(f"Pushing merged dataset to HuggingFace Hub")
    chunk_dataset.push_to_hub(
        repo_id=args.repo_id,
        split=split,
        private=True,
        toke=args.auth_token,
    )
