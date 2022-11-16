import logging
import sys
import numpy as np

from PIL import Image
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets

from BLIP.models.blip import blip_decoder
from src.precontext.utils import image_to_tensors
from src.utils import build_general_config
from tqdm import tqdm

LOG = logging.getLogger(__name__)


def main():
    config = build_general_config()

    # Set the model
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_url,
                         image_size=config.image_size,
                         vit='base')
    model.eval()

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

    def collate_batch(batch):
        img_tensors = []
        for t, (sample) in enumerate(batch):
            img_tensors.append(torch.unsqueeze(sample["pixel_values"], 0))
        images = torch.tensor((t + 1, 3, config.image_size, config.image_size),
                              dtype=torch.float)
        torch.cat(img_tensors, out=images)
        return images.to(config.device)

    for t, (start, stop) in enumerate(zip(chunks[:-1], chunks[1:])):
        # Check if the chunk was already processed
        if t < config.num_chunks_processed:
            continue
        LOG.info(f"Loading chunk from {start}% to {stop}%")

        # Loading the dataset chunk
        dataset = load_dataset(
            "Maysee/tiny-imagenet",
            split=f"{config.split}[{int(start)}%:{int(stop)}%]")
        dataset.set_transform(image_to_tensors)
        dataloader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                collate_fn=collate_batch)
        LOG.info(f"Starting caption generation")
        captions = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch_captions = model.generate(batch,
                                                sample=False,
                                                num_beams=3,
                                                max_length=20,
                                                min_length=5)
                captions.extend(batch_captions)
        dataset.add_column("caption", captions)
        dataset.remove_columns("pixel_values")
        LOG.info(f"Dataset chunk processing finished")
        LOG.info(f"Pushing chunk to HF respository: {config.repo_id}")
        repo_split = f"{config.split}_{t}"
        LOG.info(f"Storing the chunk using the split: {repo_split}")
        dataset.push_to_hub(
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