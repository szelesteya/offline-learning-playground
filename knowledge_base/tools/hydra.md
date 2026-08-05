# Hydra

Config management + CLI library used to compose the agent/env/training setup for a run
from separate YAML files and override any of it from the command line.

## `@hydra.main` entrypoint

Decorator that turns a function into a config-driven entrypoint: it loads and composes the
config *before* calling the function, and injects the result as the function's single
argument.

**In this repo:** [`src/olp/cli/train.py:33`](../../src/olp/cli/train.py#L33)
```python
@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
```
`config_path` is resolved relative to the file the decorator is in (not the cwd), and
`config_name` is the top-level file to start composing from
([`configs/config.yaml`](../../configs/config.yaml)).

**Docs:** https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/

## Config groups

A directory of interchangeable YAML files (e.g. `agent/model_free/cql.yaml` vs.
`agent/model_based/mopo.yaml`) that all fill the same slot in the composed config. Only one
file per group is active in a given run, selected by directory path.

**In this repo:** [`configs/agent/`](../../configs/agent/), [`configs/env/`](../../configs/env/),
[`configs/training/`](../../configs/training/) are three independent groups. Switched from the
command line:
```
uv run python -m olp.cli.train agent=model_based/mopo env=minari/pointmaze_umaze
```

**Docs:** https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/

## `_target_` + `hydra.utils.instantiate`

A config can name a Python class via `_target_`; `hydra.utils.instantiate(cfg)` imports that
class and calls it with every other key in the config as keyword arguments. The config
selects *behavior*, not just data — no hand-written if/elif registry mapping names to
classes.

**In this repo:** [`configs/env/minari/fourrooms.yaml`](../../configs/env/minari/fourrooms.yaml)
```yaml
_target_: olp.envs.minari_source.MinariEnvSource
name: fourrooms
dataset_id: D4RL/minigrid/fourrooms-v0
```
built in [`cli/env.py`](../../src/olp/cli/env.py) with
`source = hydra.utils.instantiate(cfg.env)` — every other key becomes a constructor kwarg to
`MinariEnvSource.__init__`. Adding a second backend is just a new class + a new YAML file,
nothing in `env.py` changes.

**Docs:** https://hydra.cc/docs/advanced/instantiate_objects/overview/

## The `defaults` list

Declares which option from each config group is active by default, and in what order they're
merged (later entries override earlier ones). `_self_` controls where the current file's own
keys slot into that merge order.

**In this repo:** [`configs/config.yaml`](../../configs/config.yaml)
```yaml
defaults:
  - agent: model_free/cql
  - env: minari/fourrooms
  - training: default
  - _self_
```

**Docs:** https://hydra.cc/docs/tutorials/basic/your_first_app/defaults/

## Multiple top-level configs, one set of groups

`config_name` just picks which top-level YAML to start composing from — different entrypoints
can reuse the same config groups (`agent/`, `env/`) in different combinations without
duplicating them.

**In this repo:** [`configs/config.yaml`](../../configs/config.yaml) (used by
[`train.py`](../../src/olp/cli/train.py)) and [`configs/eval.yaml`](../../configs/eval.yaml)
(used by [`eval.py`](../../src/olp/cli/eval.py)) both pull from `agent/` and `env/`, but
`eval.yaml` skips the `training` group and adds its own `checkpoint_path`/`eval_episodes`.

**Docs:** https://hydra.cc/docs/tutorials/basic/your_first_app/composition/

## Config names and config *group* names are separate namespaces

A top-level config file (`configs/env.yaml`, resolved via `config_name="env"`) and a config
group directory of the same name (`configs/env/`, resolved via a group reference like
`env: minari/fourrooms`) do not collide — Hydra checks them with two different predicates
(`is_config` = "is this a file", `is_group` = "is this a directory") depending on whether it's
resolving the primary config or a defaults-list entry. Hydra ships its own `hydra/env` group
(job environment variables) as a working precedent for this exact coexistence.

**In this repo:** [`configs/env.yaml`](../../configs/env.yaml) (the top-level config behind
`olp env`) sits next to [`configs/env/`](../../configs/env/) (the group `agent`/`env`/`training`
all pull from). `uv run python -m olp.cli.env --info config` shows both resolving correctly
in the same defaults tree.

**Docs:** https://hydra.cc/docs/advanced/search_path/

## Mandatory values (`???`)

A field set to `???` has no default — reading it raises `MissingMandatoryValue` unless a
command-line override supplies one. Fails loudly instead of silently running with a bogus
default.

**In this repo:** [`configs/eval.yaml`](../../configs/eval.yaml) `checkpoint_path: ???`, enforced
in [`eval.py:20`](../../src/olp/cli/eval.py#L20) (`_ = cfg.checkpoint_path`):
```
uv run olp eval                                    # -> MissingMandatoryValue
uv run olp eval checkpoint_path=outputs/cql_gridworld/2026-08-02_13-02-31   # -> OK
```

**Docs:** https://omegaconf.readthedocs.io/en/latest/usage.html#mandatory-values

## Variable interpolation

`${...}` references another value in the composed config tree, resolved at read time — lets
one group's config depend on another's without hardcoding.

**In this repo:** [`configs/config.yaml`](../../configs/config.yaml) builds an experiment name
from the active agent and env (`experiment_name: ${agent.name}_${env.name}`) — a group
referencing a value defined outside itself.

**Docs:** https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation

## The `hydra:` resolver

A special interpolation namespace (distinct from the app's own config tree) exposing Hydra's
*own* runtime state — e.g. which physical directory it decided to run this job in, or the
job/sweep number. Reusing it instead of reconstructing a path yourself keeps app config and
Hydra's actual behavior from silently drifting apart (see history: an earlier version of
`training.log_dir` built its own `outputs/${experiment_name}` path independently of
`hydra.run.dir` below, so the two disagreed about where files actually landed).

**In this repo:** [`configs/training/default.yaml`](../../configs/training/default.yaml)
```yaml
log_dir: ${hydra:runtime.output_dir}
```
always equals whatever directory Hydra is actually writing `.hydra/config.yaml` and `train.log`
to for this job.

**Docs:** https://hydra.cc/docs/configure_hydra/intro/#accessing-hydras-configuration

## Command-line overrides

Any leaf in the composed config can be set from the CLI with a dotted path, no code changes
required.

**In this repo:**
```
uv run python -m olp.cli.train training.total_steps=500 seed=42
```

**Docs:** https://hydra.cc/docs/advanced/override_grammar/basic/

## Multirun (`-m`)

Runs the entrypoint once per combination when a group override is given a comma-separated
list — a built-in grid sweep, no extra looping code.

**In this repo:**
```
uv run python -m olp.cli.train -m agent=model_free/cql,model_based/mopo env=minari/fourrooms,minari/pointmaze_umaze
```
runs all 4 `agent`×`env` combinations (`pointmaze_umaze` needs `uv sync --extra mujoco`).

**Docs:** https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

## Per-run output directory

Hydra changes the working directory for each run to a fresh folder (pattern configurable via
`hydra.run.dir` / `hydra.sweep.dir` + `hydra.sweep.subdir`, both interpolatable like any other
config value) and dumps the fully-resolved config + logs there, so every run is reproducible
from its own output dir alone.

**In this repo:** [`configs/config.yaml`](../../configs/config.yaml) keys the pattern off our
own `experiment_name` instead of Hydra's date/time-only default, so runs are easy to find:
```yaml
hydra:
  run:
    dir: outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.num}_${experiment_name}
```
→ `outputs/cql_fourrooms/2026-08-02_13-02-31/.hydra/config.yaml` + `train.log` per run, or
`multirun/<timestamp>/<job#>_<experiment_name>/...` under `-m`. Not committed — see
[`.gitignore`](../../.gitignore).

**Docs:** https://hydra.cc/docs/configure_hydra/intro/

## Read-only commands: opting out of run-directory side effects

`@hydra.main(version_base=None, ...)` defaults `hydra.job.chdir` to `True` (with a deprecation
warning) — every job changes the process's working directory to its output dir. For a command
that only inspects things and shouldn't scatter folders on every invocation, three settings
turn that off: `hydra.job.chdir: false` (don't chdir), `hydra.run.dir: .` (nowhere to chdir to
even if something tries), `hydra.output_subdir: null` (skip writing `.hydra/config.yaml`).

**In this repo:** [`configs/env.yaml`](../../configs/env.yaml) — `olp env` is read-only and
leaves the working tree untouched (`git status` stays clean across runs), unlike `train`/`eval`
which intentionally do want per-run output dirs.

**Docs:** https://hydra.cc/docs/configure_hydra/workdir/
