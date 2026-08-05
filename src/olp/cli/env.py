"""Inspect the environment/dataset selected by the `env` config group.

Read-only: loads the dataset (downloading it on first use), reports its metadata, recovers
the evaluation environment, and samples a few episodes. No training/eval pipeline - see
adrs/0001-env-is-eval-only.md for why the environment is only ever used this way.

Examples
--------
    uv run olp env
    uv run olp env env=minari/pointmaze_umaze
    uv run olp env sample_episodes=5 seed=1
"""

import sys

import gymnasium as gym
import hydra
from omegaconf import DictConfig

from olp.envs.base import EnvBackendError


def _describe_space_match(dataset_space: gym.Space, env_space: gym.Space) -> str:
    if dataset_space == env_space:
        return "match"
    if isinstance(dataset_space, gym.spaces.Dict) and isinstance(env_space, gym.spaces.Dict):
        diffs = [
            f"{key}: dataset={dataset_space.spaces[key]!r} vs env={env_space.spaces.get(key)!r}"
            for key in dataset_space.spaces
            if dataset_space.spaces[key] != env_space.spaces.get(key)
        ]
        return (
            "differs on " + "; ".join(diffs) + " (often a serialization artifact - custom "
            "space subclasses, e.g. MiniGrid's MissionSpace, are stored as their base type)"
        )
    return f"differs: dataset={dataset_space!r} vs env={env_space!r}"


def _obs_summary(observations) -> str:
    if isinstance(observations, dict):
        return ", ".join(f"{k}: {len(v)}" for k, v in observations.items())
    return str(len(observations))


@hydra.main(version_base=None, config_path="../../../configs", config_name="env")
def main(cfg: DictConfig) -> None:
    print("== Selection ==")
    print(f"  name:        {cfg.env.name}")
    print(f"  target:      {cfg.env._target_}")
    print(f"  dataset_id:  {cfg.env.dataset_id}")

    try:
        source = hydra.utils.instantiate(cfg.env)
    except Exception as exc:
        print(f"\n== Dataset ==\n  could not build env source: {exc}")
        sys.exit(1)

    try:
        spec = source.spec()
    except EnvBackendError as exc:
        print(f"\n== Dataset ==\n  {exc}")
        sys.exit(1)

    print("\n== Dataset ==")
    print(f"  total_episodes:  {spec.total_episodes}")
    print(f"  total_steps:     {spec.total_steps}")
    print(f"  observation_space: {spec.observation_space}")
    print(f"  action_space:      {spec.action_space}")
    has_ref_scores = spec.ref_min_score is not None and spec.ref_max_score is not None
    print(f"  normalized score reference: {'available' if has_ref_scores else 'unavailable (ref_min/max_score not set)'}")

    try:
        env = source.make_eval_env(seed=cfg.seed)
    except EnvBackendError as exc:
        print(f"\n== Eval environment ==\n  {exc}")
    else:
        print("\n== Eval environment ==")
        print(f"  id:                {env.spec.id}")
        print(f"  max_episode_steps: {env.spec.max_episode_steps}")
        print(f"  observation_space match: {_describe_space_match(spec.observation_space, env.observation_space)}")
        print(f"  action_space match:      {_describe_space_match(spec.action_space, env.action_space)}")
        env.close()

    try:
        dataset = source.load_dataset()
        dataset.set_seed(cfg.seed)
        episodes = list(dataset.sample_episodes(cfg.sample_episodes))
    except EnvBackendError as exc:
        print(f"\n== Sample episodes ==\n  {exc}")
        return

    print(f"\n== Sample episodes (seed={cfg.seed}) ==")
    print("  note: observations has N+1 entries per key (initial + final obs), actions has N")
    for ep in episodes:
        n_actions = len(ep.actions)
        ep_return = float(ep.rewards.sum())
        print(
            f"  episode {ep.id}: actions={n_actions} return={ep_return:.3f} "
            f"terminated={bool(ep.terminations[-1])} truncated={bool(ep.truncations[-1])} "
            f"observations=({_obs_summary(ep.observations)})"
        )


if __name__ == "__main__":
    main()
