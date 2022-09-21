import argparse
import logging
import torch

#from datasets import load_dataset, concatenate_datasets
from diffusers import StableDiffusionPipeline

LOG = logging.getLogger(__name__)


def generate_with_diffuser(batch, diffuser: StableDiffusionPipeline,
                           latents: torch.tensor, generator: torch.Generator,
                           args: argparse.Namespace):
    with torch.autocast(args.device):
        image = diffuser(
            batch["sentence"],
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
                batch["sentence"],
                guidance_scale=7.5,
                latents=latents,
            )
        if idx == args.max_nsfw_tries:
            LOG.info("NSFW detected and max number of tries reached")
            LOG.info("Image set to black")
            break
    generator.manual_seed(args.seed)
    batch["image"] = image["sample"][0]
    for k, v in batch.items():
        LOG.info(f"{k}: {v}")
    return batch
