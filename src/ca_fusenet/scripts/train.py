'''
Docstring for ca_fusenet.scripts.train

This script serves as the main entry point for training the CA-FuseNet model. It sets up logging, handles configuration using Hydra,
and initializes the training environment by setting the random seed and ensuring the output directory exists.
'''
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from ca_fusenet.utils.logging import setup_logging
from ca_fusenet.utils.paths import ensure_dir
from ca_fusenet.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = logging.getLogger("ca_fusenet.train")

    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    print(resolved)

    set_seed(cfg.seed, deterministic=cfg.training.deterministic)
    output_dir = ensure_dir(cfg.paths.output_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info("startup ok")


if __name__ == "__main__":
    main()
