"""CLI tests for the preprocess app (plan 2.1, T4b)."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import typer
from typer.testing import CliRunner

from apps.preprocess import main as preprocess_main
import typer
from typer.testing import CliRunner

# The shipped CLI registers ``preprocess.main`` directly on the root app
# (src/main.py); replicate that so ORIGIN parsing behaves the same.
_root = typer.Typer(pretty_exceptions_enable=False)
_root.command(name="preprocess")(preprocess_main)


@_root.command(name="_other")  # ensures the root parses as a group, as in src/main.py
def _other() -> None:
    raise NotImplementedError
app = _root

runner = CliRunner()


def test_typer_preserves_repeated_option_order() -> None:
    """Typer must deliver repeated --op values in flag order for _parse_ops."""
    probe = typer.Typer()

    @probe.command()
    def go(ops: list[str] = typer.Option([], "--op")) -> None:
        print("\n".join(ops))

    result = runner.invoke(probe, ["--op", "clean:a", "--op", "dropna:b", "--op", "embed:c"])
    assert result.exit_code == 0
    assert result.output.splitlines() == ["clean:a", "dropna:b", "embed:c"]


def _make_fixture(tmp_path) -> tuple[str, str]:
    src = tmp_path / "src"
    src.mkdir()
    pl.DataFrame(
        {
            "text": ["hello world", "hi", None, "valid text here"],
            "language": ["en", "en", "en", "en"],
            "is_license_safe": [True, True, True, True],
            "publication_date": [datetime(2020, 1, 1)] * 4,
            "field_id": [1, 2, 3, 4],
            "type": ["article"] * 4,
        }
    ).write_parquet(src / "part_0.parquet")
    return str(src), str(tmp_path)


def test_cli_dry_run_clean_dropna(tmp_path) -> None:
    origin, base_dir = _make_fixture(tmp_path)
    result = runner.invoke(
        app,
        [
            "preprocess",
            origin,
            "--op",
            "clean:text",
            "--trim-min-chars",
            "5",
            "--op",
            "dropna:text,language",
            "--dry-run",
            "--source-base-dir",
            base_dir,
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # dry-run prints the collected frame
    assert "shape:" in result.output
    assert "hello world" in result.output


def test_cli_embedder_without_embed_op_is_bad_parameter(tmp_path) -> None:
    origin, base_dir = _make_fixture(tmp_path)
    result = runner.invoke(
        app,
        [
            "preprocess",
            origin,
            "--op",
            "clean:text",
            "--op-embedder",
            "modernbert-base",
            "--source-base-dir",
            base_dir,
        ],
    )
    assert result.exit_code != 0
    assert "no embed" in result.output
    assert "Traceback" not in result.output
