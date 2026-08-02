"""Dummy evaluation entrypoint - prints the composed config, no real eval loop yet.

Examples
--------
    uv run olp eval checkpoint_path=outputs/cql_gridworld/2026-08-02_13-02-31
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    log.info("Composed eval config for experiment '%s'", cfg.experiment_name)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    _ = cfg.checkpoint_path  # raises MissingMandatoryValue if not overridden


if __name__ == "__main__":
    main()
