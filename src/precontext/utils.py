import torch


def generate_with_diffuser(sample, diffuser, latents, generator, args):
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
            break
    generator.manual_seed(args.seed)
    sample["pixel_array"] = image["sample"][0]
    return sample