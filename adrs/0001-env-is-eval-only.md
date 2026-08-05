# 1. The environment is used for dataset provenance and evaluation only

## Context

Offline RL trains a policy from a fixed, previously-collected dataset with no environment
interaction during training — that's the defining constraint of the setting (Levine, Kumar,
Tucker, Fu, 2020, [arXiv:2005.01643](https://arxiv.org/abs/2005.01643)). But the environment
doesn't disappear entirely: every major offline RL benchmark still uses a simulator to
*evaluate* the resulting policy by rolling it out and measuring return, because that's the
cheap, low-variance way to know if training worked. D4RL
([arXiv:2004.07219](https://arxiv.org/abs/2004.07219)) and RL Unplugged
([arXiv:2006.13888](https://arxiv.org/abs/2006.13888)) both follow this split: dataset for
training, simulator for eval.

The alternative — never touching the environment, not even for evaluation — is a distinct,
harder subfield called Off-Policy Evaluation (OPE), with its own benchmark
(DOPE, [arXiv:2103.16596](https://arxiv.org/abs/2103.16596)) and its own motivation: using the
environment to pick hyperparameters or checkpoints "breaks the offline assumption" when no
simulator is actually available (Paine et al., 2020,
[arXiv:2007.09055](https://arxiv.org/abs/2007.09055)).

This repo targets the standard benchmark protocol (a simulator is available), not the
OPE-only setting.

## Decision

The environment interface (`olp.envs.EnvSource`) splits dataset-derived facts from
environment construction:

- `spec()` reads only the dataset's recorded metadata (episode/step counts, observation and
  action spaces). It never constructs an environment, so it works even when the simulator
  backing a given dataset isn't installed (e.g. inspecting a MuJoCo-based dataset without
  `mujoco` installed).
- `make_eval_env()` is the only method that builds a real environment, and it exists solely
  for evaluation rollouts. No agent or training code calls it during training.

## Consequences

- Dataset inspection (`olp env`) degrades gracefully: it can report episode counts, spaces,
  and sample transitions from a dataset whose simulator dependency is missing, and only the
  eval-environment section of the report fails.
- Training code has no path to step the environment even by accident — the type signature
  itself enforces the offline constraint, rather than relying on convention.
- If a genuinely simulator-free (OPE) evaluation path is added later, it plugs in as an
  alternative to `make_eval_env()`-based rollout evaluation without touching `spec()` or the
  training path at all.
