from hydra.utils import instantiate
from omegaconf import OmegaConf

def build_general_config():
    cli_config = OmegaConf.from_cli()
    if "config" not in cli_config:
        raise ValueError(
            "Please pass 'config' to specify configuration yaml file for running Precontext"
        )
    conf_yaml = OmegaConf.load(cli_config.config)
    conf = instantiate(conf_yaml)
    cli_config.pop("config")
    return OmegaConf.merge(conf, cli_config)
