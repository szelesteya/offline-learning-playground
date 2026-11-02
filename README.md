# Offline Learning Playground

A research playground for offline reinforcement learning, with a particular focus on
**model-based offline RL**. It combines three things in one repo:

- **Knowledge base** — research references organized as markdown notes, grouped by topic.
- **Code** — a modular library of agents, environments, and benchmark adapters driven by a
  single CLI.
- **Experiments** — configs, notes, and results from training runs.

## Repository structure

```
docs/                MkDocs documentation
  adrs/              Architecture decision records for this repo's own design choices
  knowledge_base/    Research references, reading notes, and tool deep-dives

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

## Documentation

This repository includes an interactive MkDocs documentation site under `docs/`:

- **ADRs**: Architecture decision records under `docs/adrs/`.
- **Knowledge Base**: Research papers, notes, and tool deep-dives under `docs/knowledge_base/`.

- Live preview: uv run mkdocs serve
- Static build: uv run mkdocs build
- Regenerate navigation after adding files or folders: uv run python scripts/update_nav.py
