from __future__ import annotations

from typing import TYPE_CHECKING

import minari

from olp.envs.base import EnvBackendError, EnvSpec

if TYPE_CHECKING:
    import gymnasium as gym
    from minari import MinariDataset


class MinariEnvSource:
    """EnvSource backed by a Farama Minari dataset (the maintained D4RL successor)."""

    def __init__(
        self,
        name: str,
        dataset_id: str,
        download: bool = True,
        ref_min_score: float | None = None,
        ref_max_score: float | None = None,
    ) -> None:
        self.name = name
        self.dataset_id = dataset_id
        self.download = download
        self.ref_min_score = ref_min_score
        self.ref_max_score = ref_max_score
        self._dataset: MinariDataset | None = None

    def load_dataset(self) -> MinariDataset:
        if self._dataset is not None:
            return self._dataset
        try:
            self._dataset = minari.load_dataset(self.dataset_id, download=self.download)
        except FileNotFoundError as exc:
            local = ", ".join(sorted(minari.list_local_datasets())) or "(none)"
            raise EnvBackendError(
                f"Dataset '{self.dataset_id}' is not available locally (download=False).\n"
                f"Run: uv run minari download {self.dataset_id}\n"
                f"Locally available: {local}"
            ) from exc
        except ValueError as exc:
            raise EnvBackendError(
                f"Could not load/download dataset '{self.dataset_id}': {exc}\n"
                f"Check the id with: uv run minari list remote"
            ) from exc
        return self._dataset

    def spec(self) -> EnvSpec:
        dataset = self.load_dataset()
        return EnvSpec(
            name=self.name,
            backend="minari",
            dataset_id=self.dataset_id,
            observation_space=dataset.observation_space,
            action_space=dataset.action_space,
            total_episodes=dataset.total_episodes,
            total_steps=dataset.total_steps,
            ref_min_score=self.ref_min_score,
            ref_max_score=self.ref_max_score,
        )

    def make_eval_env(self, seed: int | None = None, **kwargs) -> gym.Env:
        dataset = self.load_dataset()
        try:
            env = dataset.recover_environment(eval_env=True, **kwargs)
        except ValueError as exc:
            raise EnvBackendError(
                f"Dataset '{self.dataset_id}' has no recorded environment spec: {exc}"
            ) from exc
        except Exception as exc:  # gym.make can fail in many backend-specific ways
            raise EnvBackendError(
                f"Could not construct the environment for '{self.dataset_id}': {exc}\n"
                f"If this is a MuJoCo-based task, run: uv sync --extra mujoco"
            ) from exc
        if seed is not None:
            env.reset(seed=seed)
        return env
