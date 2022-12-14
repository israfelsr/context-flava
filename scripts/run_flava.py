import logging
import sys

import datasets
import transformers

from flava.data.datamodules import MMDataModule
from flava.definitions import FLAVAArguments
from flava.model import FLAVAClassificationLightningModule
from flava.utils import build_config, build_datamodule_kwargs
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

LOG = logging.getLogger(__name__)


def main():
    wandb_logger = WandbLogger()
    config: FLAVAArguments = build_config()
    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    LOG.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Setting up the datamodule
    assert len(config.datasets.selected) == 1
    datamodule = MMDataModule(
        **build_datamodule_kwargs(config.datasets.vl, config.training), )

    datamodule.setup("fit")

    model = FLAVAClassificationLightningModule(
        num_classes=config.datasets.num_classes,
        learning_rate=config.training.learning_rate,
        adam_eps=config.training.adam_eps,
        adam_weight_decay=config.training.adam_weight_decay,
        adam_betas=config.training.adam_betas,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.lightning.max_steps,
        **config.model,
    )

    callbacks = [LearningRateMonitor(logging_interval="step")]

    if config.training.lightning_checkpoint is not None:
        callbacks.append(
            ModelCheckpoint(**OmegaConf.to_container(
                config.training.lightning_checkpoint)))

    trainer = Trainer(**OmegaConf.to_container(config.training.lightning),
                      callbacks=callbacks,
                      logger=wandb_logger,
                      max_epochs=5)

    ckpt_path = config.training.lightning_load_from_checkpoint
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    trainer.validate(datamodule=datamodule)


if __name__ == "__main__":
    main()
