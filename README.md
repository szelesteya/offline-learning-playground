# Offline Learning Playground

A research playground for offline reinforcement learning, with a particular focus on
**model-based offline RL**. It combines three things in one repo:

- **Knowledge base** — research references organized as markdown notes, grouped by topic.
- **Code** — a modular library of agents, environments, and benchmark adapters driven by a
  single CLI.
- **Experiments** — configs, notes, and results from training runs.

## Repository structure

```
knowledge_base/     Research references as markdown, grouped by topic (structure TBD)
adrs/                Architecture decision records for this repo's own design choices

src/olp/             Importable package
  cli/                Single Typer entrypoint (`olp train`, `olp eval`, `olp env`, ...)
  agents/             Model-free and model-based agent implementations
  envs/               Environment/dataset sources (Minari, ...) - the offline benchmark interface
  data/               Dataset loading, replay buffers
  training/           Generic trainer, evaluator, callbacks
  models/             Shared neural network building blocks
  utils/              Logging, seeding, metrics

configs/             Hydra config groups (agent / env / training) composed at the CLI
experiments/         Dated run configs, notes, and results
tests/               Unit tests
scripts/             One-off utility scripts
```

## Goals

- Keep a living, organized map of offline RL literature, especially model-based methods.
- Implement agents and simple environments in a modular way so new algorithms and envs can
  be dropped in without touching unrelated code.
- Drive everything — agent, environment, hyperparameters, training setup — from one CLI, with
  Hydra-composed configs so runs are reproducible and easy to sweep.
- Benchmark against standard offline RL suites as goto references for evaluating new ideas.

## Getting started

```bash
uv sync
uv run olp --help
```

## Status

Early scaffolding — knowledge base and code structure are being built out.