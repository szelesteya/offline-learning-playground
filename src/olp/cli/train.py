"""Dummy training entrypoint used to shake out the project's Hydra config layout.

There is no real agent/environment/training loop yet - this only resolves and
prints the composed config, so the config groups (agent / env / training) can
be iterated on independently of the actual implementations.

Examples
--------
Run with the defaults declared in configs/config.yaml:
    uv run python -m olp.cli.train

Swap a config group from the command line:
    uv run python -m olp.cli.train agent=model_based/mopo env=simple/pendulum

Override individual leaf values (dotted path into the composed config):
    uv run python -m olp.cli.train training.total_steps=500 seed=42

Sweep multiple runs in one invocation (Hydra multirun):
    uv run python -m olp.cli.train -m agent=model_free/cql,model_based/mopo env=simple/gridworld,simple/pendulum

Each run gets its own working directory (outputs/<date>/<time>/ by default)
containing the fully resolved config Hydra used, under .hydra/config.yaml.
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Composed config for experiment '%s'", cfg.experiment_name)
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
