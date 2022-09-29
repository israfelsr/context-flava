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
    while any(image["nsfw_content_detected"]):
        idx += 1
        if idx == args.max_nsfw_tries:
            LOG.info("NSFW detected and max number of tries reached")
            if args.set_to_none:
                nsfw_idx = [
                    i for i, x in enumerate(image["nsfw_content_detected"])
                    if x
                ]
                image["sample"][nsfw_idx] = None
                LOG.info("Image set to None")
            else:
                LOG.info("Image set to black")
            break
        generator = generator.manual_seed(generator.seed() + 1)
        latents = torch.randn((1, diffuser.unet.in_channels, 64, 64),
                              generator=generator,
                              device="cuda")
        with torch.autocast("cuda"):
            image = diffuser(
                batch["sentence"],
                guidance_scale=7.5,
                latents=latents,
            )
    generator.manual_seed(args.seed)
    batch["image"] = image["sample"]
    return batch
