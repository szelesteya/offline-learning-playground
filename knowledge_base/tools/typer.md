# Typer

CLI framework used for the top-level `olp` command. Only a thin dispatcher here — it does not
wrap Hydra's own argv parsing (see below), it just routes to a subprocess per subcommand.

## App object + `@app.command()`

`typer.Typer()` creates an app; each `@app.command()`-decorated function becomes a subcommand
named after the function.

**In this repo:** [`src/olp/cli/app.py`](../../src/olp/cli/app.py)
```python
app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.command(context_settings=_PASSTHROUGH)
def train(ctx: typer.Context) -> None: ...
```
→ `olp train`, `olp eval`.

**Docs:** https://typer.tiangolo.com/tutorial/subcommands/

## Passthrough args (`allow_extra_args` + `ignore_unknown_options`)

By default Click (which Typer sits on) parses every argument itself and rejects anything it
doesn't recognize as an option. These two `context_settings` disable that, so unrecognized
arguments are collected verbatim in `ctx.args` instead of erroring — necessary here because
Hydra's `key=value` overrides and `-m` flag aren't things Typer/Click understand, and Hydra
needs to see them unmodified.

**In this repo:** [`src/olp/cli/app.py:22-30`](../../src/olp/cli/app.py#L22-L30)
```python
_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}

def _dispatch(module: str, extra_args: list[str]) -> None:
    result = subprocess.run([sys.executable, "-m", module, *extra_args])
```
So `olp train agent=model_based/mopo` forwards `["agent=model_based/mopo"]` straight through to
`python -m olp.cli.train`, which parses it as a normal Hydra override.

**Docs:** https://click.palletsprojects.com/en/stable/advanced/#forwarding-unknown-options

## Console script entry point

`[project.scripts]` in `pyproject.toml` maps a shell command to a Python callable, installed by
`uv sync`/`pip install`. Not a Typer feature itself (any callable works), but it's what makes
`olp` an actual command instead of `python -m olp.cli.app`.

**In this repo:** [`pyproject.toml`](../../pyproject.toml)
```toml
[project.scripts]
olp = "olp.cli.app:main"
```

**Docs:** https://packaging.python.org/en/latest/specifications/entry-points/#use-for-scripts
