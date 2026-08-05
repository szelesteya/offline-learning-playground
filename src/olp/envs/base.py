"""Environment/dataset source interface.

In offline RL the environment is never stepped during training - only for dataset
provenance and evaluation rollouts (see adrs/0001-env-is-eval-only.md). `spec()` reflects
that: it reads dataset metadata only and never constructs an environment, so inspection
still works even if the simulator backing `make_eval_env()` isn't installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import gymnasium as gym


class EnvBackendError(RuntimeError):
    """Dataset or eval environment could not be provided; message is user-actionable."""


@dataclass(frozen=True, slots=True)
class EnvSpec:
    """Dataset-derived facts about an offline RL benchmark task."""

    name: str
    backend: str
    dataset_id: str
    observation_space: gym.Space
    action_space: gym.Space
    total_episodes: int
    total_steps: int
    ref_min_score: float | None = None
    ref_max_score: float | None = None


class EnvSource(Protocol):
    """One offline dataset plus the environment used to evaluate policies on it."""

    name: str

    def spec(self) -> EnvSpec: ...

    def load_dataset(self) -> Any:
        """Backend-native dataset object (e.g. a `minari.MinariDataset`)."""
        ...

    def make_eval_env(self, seed: int | None = None) -> gym.Env: ...


def normalized_score(spec: EnvSpec, score: float) -> float | None:
    """D4RL convention: 100 * (score - ref_min) / (ref_max - ref_min).

    Returns None if the task has no reference scores (most tasks outside the original
    D4RL suite don't - guessing constants would be worse than reporting unavailable).
    """
    if spec.ref_min_score is None or spec.ref_max_score is None:
        return None
    return 100.0 * (score - spec.ref_min_score) / (spec.ref_max_score - spec.ref_min_score)
