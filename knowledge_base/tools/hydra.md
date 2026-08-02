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
uv run python -m olp.cli.train agent=model_based/mopo env=simple/pendulum
```

**Docs:** https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/

## The `defaults` list

Declares which option from each config group is active by default, and in what order they're
merged (later entries override earlier ones). `_self_` controls where the current file's own
keys slot into that merge order.

**In this repo:** [`configs/config.yaml`](../../configs/config.yaml)
```yaml
defaults:
  - agent: model_free/cql
  - env: simple/gridworld
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
uv run python -m olp.cli.train -m agent=model_free/cql,model_based/mopo env=simple/gridworld,simple/pendulum
```
runs all 4 `agent`×`env` combinations.

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
→ `outputs/cql_gridworld/2026-08-02_13-02-31/.hydra/config.yaml` + `train.log` per run, or
`multirun/<timestamp>/<job#>_<experiment_name>/...` under `-m`. Not committed — see
[`.gitignore`](../../.gitignore).

**Docs:** https://hydra.cc/docs/configure_hydra/intro/
