# Minari

Farama's maintained dataset API for offline RL — the successor to D4RL, which is unmaintained
and needs `mujoco-py` (won't install on Windows). Minari hosts reproductions of the D4RL
datasets plus newer ones (MiniGrid, Atari via `RL Unplugged`-style tasks).

## Dataset namespaces

Dataset IDs are `<namespace>/<task>/<variant>-v<N>`. The namespace tells you the lineage and,
usually, the dependencies: `D4RL/*` are reproductions of the original D4RL suite (e.g.
`D4RL/pointmaze/umaze-v2`, `D4RL/minigrid/fourrooms-v0`); `mujoco/*` are newer MuJoCo
locomotion tasks (`mujoco/halfcheetah/expert-v0`); `minigrid/*` covers the BabyAI tasks.
**IDs are not guessable from docs/blog posts** — verify with `minari.list_remote_datasets()`
before hardcoding one (`minigrid/fourrooms-v0` looked plausible but doesn't exist; the real
id is `D4RL/minigrid/fourrooms-v0`).

**In this repo:** [`configs/env/minari/`](../../configs/env/minari/) — one YAML per dataset,
each carrying a verified `dataset_id`.

**Docs:** https://minari.farama.org/main/content/dataset_standards/

## `load_dataset(dataset_id, download=)`

Loads a dataset from `~/.minari/datasets/`. `download` defaults to `False` in Minari itself —
without it, a missing dataset raises `FileNotFoundError`. Downloading needs the `hf` extra
(`huggingface_hub`) and reading the on-disk HDF5 format needs the `hdf5` extra (`h5py`) *and*
`Pillow` for image observations — neither of the latter two is pulled in automatically by
any Minari extra, they're just assumed present.

**In this repo:** [`pyproject.toml`](../../pyproject.toml) `minari[hf,hdf5]` + explicit
`pillow` dependency; wrapped in
[`minari_source.py:32-49`](../../src/olp/envs/minari_source.py#L32-L49)
(`MinariEnvSource.load_dataset`), which turns `FileNotFoundError` into a message naming the
exact `minari download` command instead of a raw traceback.

**Docs:** https://minari.farama.org/main/api/minari_functions/#minari.load_dataset

## `recover_environment(eval_env=)`

Rebuilds the Gymnasium environment used to generate the dataset, from a `gymnasium.EnvSpec`
stored in the dataset's metadata (so `dataset.env_spec` is available with **no environment
construction** - the split `spec()` (metadata only) / `make_eval_env()` (constructs a real
env) in this repo relies on that). `eval_env=True` does **not** raise when a dataset has no
separate eval spec — it silently falls back to the training env spec and just logs an info
message. It only fails when the underlying `gym.make()` needs a package that isn't installed
(e.g. a MuJoCo task without `mujoco` installed).

**In this repo:**
[`minari_source.py:65-80`](../../src/olp/envs/minari_source.py#L65-L80)
(`MinariEnvSource.make_eval_env`) catches that failure and points at
`uv sync --extra mujoco`.

**Docs:** https://minari.farama.org/main/api/minari_dataset/minari_dataset/#minari.dataset.minari_dataset.MinariDataset.recover_environment

## `EpisodeData` and the N+1 observations convention

Iterating/sampling episodes yields `EpisodeData(id, observations, actions, rewards,
terminations, truncations, infos)`. `observations` has **one more entry than `actions`**
per episode — it includes the initial reset observation and the final one, while `actions`
has exactly one entry per transition taken.

**In this repo:** [`cli/env.py`](../../src/olp/cli/env.py) prints `actions=N` next to each
key's observation count (`N+1`) for every sampled episode, to make the convention visible
rather than asserted.

**Docs:** https://minari.farama.org/main/api/minari_dataset/episode_data/

## `sample_episodes` / `iterate_episodes` / `set_seed`

`dataset.set_seed(seed)` makes subsequent `sample_episodes(n)` calls deterministic (same
episode ids across runs); `iterate_episodes(episode_indices=...)` walks a specific, ordered
subset instead of a random sample.

**In this repo:** [`cli/env.py`](../../src/olp/cli/env.py) — `dataset.set_seed(cfg.seed)`
then `dataset.sample_episodes(cfg.sample_episodes)`.

**Docs:** https://minari.farama.org/main/content/basic_usage/

## Stored spaces can disagree with the live environment's spaces

A dataset's `observation_space`/`action_space` are the JSON-serialized spaces recorded at
collection time. Custom `gymnasium.Space` subclasses don't round-trip: MiniGrid's
`MissionSpace` (a function that generates mission strings) is stored as a plain `Text` space,
so `dataset.observation_space != env.observation_space` even though nothing is actually
wrong. Worth checking, not worth trusting blindly.

**In this repo:** [`cli/env.py`](../../src/olp/cli/env.py) `_describe_space_match()` reports
per-key differences on `Dict` spaces instead of a bare `!=`, and names the serialization
caveat directly. Observed live on `D4RL/minigrid/fourrooms-v0`.

**Docs:** https://minari.farama.org/main/content/dataset_standards/#observation-and-action-spaces

## CLI: `minari list` / `minari download`

`minari list remote` / `minari list local` enumerate available vs. already-downloaded
datasets; `minari download <id>` fetches one without going through Python.

**In this repo:** the error messages in
[`minari_source.py`](../../src/olp/envs/minari_source.py) name these commands directly so a
failure is immediately actionable from the terminal.

**Docs:** https://minari.farama.org/main/content/basic_usage/#minari-cli
