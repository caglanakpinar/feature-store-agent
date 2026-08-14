"""feature-store-agent: an interactive session with this project's feature engineering helper.

    python cli.py generate

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


if __name__ == "__main__":
    cli()
