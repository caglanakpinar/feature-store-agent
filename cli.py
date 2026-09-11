"""feature-store-agent: an interactive session with this project's feature engineering helper.

    python cli.py generate      # chat, then run the pipeline
    python cli.py configure     # edit the three agent configs in a browser

Installed as the `feature-store-agent` console script (see `[tool.poetry.scripts]` in `pyproject.toml`).
`agent-builder` — the git dependency this project's agents are built on — installs its own command as a
top-level `cli` module too (`agentic-ai`'s entry point is `cli:cli`; see its `dist-info`). Both packages
placing a module at that same top-level name means whichever installs second overwrites the other's
`cli.py` in `site-packages`, and `import cli` resolves to whichever is left — this repo's own `generate`
command, confirmed by running `python -c "import cli"` from here, not the vendored one. Harmless for
this project's own use, since nothing here calls the vendored `agentic-ai` command; worth knowing if
that command stops resolving to what `agent-builder` shipped.
"""

import click

from console.run import chat


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """feature-store-agent."""


@cli.command("generate")
def generate() -> None:
    """Start an interactive session: chat, then run the pipeline once its requirements are met."""
    chat()


@cli.command("configure")
@click.option("--port", default=8787, show_default=True, help="Port to serve the editor on.")
@click.option("--no-browser", is_flag=True, help="Print the URL instead of opening a browser.")
def configure(port: int, no_browser: bool) -> None:
    """Edit the `llms:` and `agents:` blocks of the three agentic_configurations.yaml files.

    Opens a local page — bound to 127.0.0.1, nothing reachable off this machine — that lists the
    three configs (console, data_engineer, feature_engineering) with what each one drives, and edits
    the picked one's LLMs and agents in place. Everything else in the file, comments included, is
    written back untouched. `console.config_ui` has the details.
    """
    from console.config_ui import serve  # imported here: `generate` should not pay for a web server

    serve(port=port, open_browser=not no_browser)


if __name__ == "__main__":
    cli()
