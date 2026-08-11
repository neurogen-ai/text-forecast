#!/usr/bin/env python3
import tomllib
import typer

from apps import describe, engineer, eval, preprocess, train
from config.loader import available_experiments, resolve_experiment_name
from utils.get_root_dir import get_root_dir

app = typer.Typer(pretty_exceptions_enable=False)
app.add_typer(preprocess, name="preprocess")
app.add_typer(describe, name="describe")
app.add_typer(engineer, name="engineer")
app.add_typer(eval, name="eval")
# app.add_typer(chat, name="chat")
app.add_typer(train, name="train")


def _toml_experiment() -> str | None:
    try:
        path = get_root_dir(markers=("pyproject.toml",)) / "config" / "config.toml"
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return data.get("experiment")
    except Exception:
        return None


@app.callback()
def main(
    ctx: typer.Context,
    experiment: str | None = typer.Option(
        None,
        "--experiment",
        "-e",
        help="Experiment module name (overrides config/config.toml)",
    ),
) -> None:
    """Root callback: resolve the experiment name and stash it for subcommands."""
    name = resolve_experiment_name(experiment or _toml_experiment())
    if name not in available_experiments():
        raise typer.BadParameter(
            f"Unknown experiment {name!r}. Available: {available_experiments()}"
        )
    ctx.obj = {"experiment_name": name}


if __name__ == "__main__":
    app()
