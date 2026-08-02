"""Top-level `olp` CLI.

Thin dispatcher: each subcommand shells out to its own @hydra.main entrypoint
rather than calling it directly, since Hydra wants to own argv parsing itself
and that clashes with Typer parsing the same argv first.

Examples
--------
    uv run olp train agent=model_based/mopo env=simple/pendulum
    uv run olp train -m agent=model_free/cql,model_based/mopo
    uv run olp eval checkpoint_path=outputs/cql_gridworld/2026-08-02_13-02-31
"""

import subprocess
import sys

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)

_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


def _dispatch(module: str, extra_args: list[str]) -> None:
    result = subprocess.run([sys.executable, "-m", module, *extra_args])
    raise typer.Exit(result.returncode)


@app.command(context_settings=_PASSTHROUGH)
def train(ctx: typer.Context) -> None:
    """Run training."""
    _dispatch("olp.cli.train", ctx.args)


@app.command(context_settings=_PASSTHROUGH)
def eval(ctx: typer.Context) -> None:
    """Run evaluation."""
    _dispatch("olp.cli.eval", ctx.args)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
