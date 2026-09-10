from __future__ import annotations

import os
from datetime import datetime
from logging import getLogger
from pathlib import Path

import typer

from apps.source_args import (
    build_source_backend_from_cli,
    source_backend_arg,
    source_base_dir_arg,
    source_opt_arg,
    source_volume_arg,
)
from config.env import load_env
from data.preprocess.pipeline import PreprocessJob
from apps.preprocess_ops import _parse_ops
from data.sources import LocalSourceBackend, SourceBackend
from data.sources.base import DataSource
from runtime import build_runtime
from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

app = typer.Typer(pretty_exceptions_enable=False)



def _resolve_origin(
    origin: str, source_backend: SourceBackend
) -> tuple[DataSource, str]:
    """Map a CLI origin (path or dataset name) to a DataSource and base name."""
    origin_path = Path(origin)
    if origin_path.exists() and isinstance(source_backend, LocalSourceBackend):
        base_dir = source_backend.base_dir
        try:
            name = str(origin_path.relative_to(base_dir))
        except ValueError as exc:
            raise typer.BadParameter(
                f"Path {origin} is not under the local source base_dir {base_dir}"
            ) from exc
        origin_source = source_backend.get_source(name)
        base_name = origin_path.stem
    else:
        origin_source = source_backend.get_source(origin)
        base_name = origin
    return origin_source, base_name


@app.command()
def main(
    ctx: typer.Context,
    origin: str = typer.Argument(
        help="Dataset name or local path under the source base_dir",
    ),
    name: str = typer.Option(
        "",
        "--name",
        "-n",
        help='Name of exported dataset; defaults to "<origin>-preprocessed"',
    ),
    ops: list[str] = typer.Option(
        [],
        "--op",
        help=(
            'Pipeline op in "<kind>:<colspec>" form, e.g. --op "clean:title" '
            'or --op "dropna:title,abstract". Repeatable; steps run in the '
            "order given."
        ),
    ),
    lowercase: list[bool] = typer.Option(
        [],
        "--lowercase",
        help="Clean param (per clean op): lowercase text before cleaning.",
    ),
    trim_min_chars: list[int] = typer.Option(
        [],
        "--trim-min-chars",
        help='Clean param: minimum chars to keep, e.g. --trim-min-chars 20.',
    ),
    trim_max_sigma: list[float] = typer.Option(
        [],
        "--trim-max-sigma",
        help="Clean param: drop texts longer than mean + N*sigma lengths.",
    ),
    require_terminal_period: list[bool] = typer.Option(
        [],
        "--require-terminal-period",
        help="Clean param: require a terminal period after trimming.",
    ),
    lang_policy: list[str] = typer.Option(
        [],
        "--lang-policy",
        help='Clean param: language handling; one of mark|drop|off.',
    ),
    no_drop_quality: list[bool] = typer.Option(
        [],
        "--no-drop-quality",
        help="Clean param: keep low-quality rows instead of dropping them.",
    ),
    tokeniser: str | None = typer.Option(
        None,
        "--op-tokeniser",
        help='Tokenise param: tokeniser key, e.g. --op-tokeniser whitespace.',
    ),
    embedder_key: str | None = typer.Option(
        None,
        "--op-embedder",
        help='Embed param: embedder key, e.g. --op-embedder modernbert-base.',
    ),
    embedder_kwargs_json: str | None = typer.Option(
        None,
        "--op-embedder-config",
        help='Embed param: JSON object for the embedder constructor, e.g. \'{"device": "cpu"}\'.',
    ),
    n_partitions: int = typer.Option(
        0,
        "--partitions",
        "-p",
        help="Number of partitions to split the dataset into",
    ),
    rows_per_part: int = typer.Option(
        0,
        "--rows-per-part",
        "-rp",
        help="Rows per partition",
    ),
    compression_level: int = typer.Option(
        1,
        "--compression-level",
        help="zstd compression level",
    ),
    start_date: datetime | None = typer.Option(
        None,
        "--start-date",
        "-sd",
        help="Start date to filter by, inclusive",
    ),
    end_date: datetime | None = typer.Option(
        None,
        "--end-date",
        "-ed",
        help="End date to filter by, exclusive",
    ),
    field_id: list[int] = typer.Option(
        [],
        "--field-id",
        "-fid",
        help="Field id/s to include",
    ),
    languages: list[str] = typer.Option(
        [],
        "--lang",
        help="Languages to include",
    ),
    types: list[str] = typer.Option(
        [],
        "--type",
        help="Types of document (eg. article, book chapter) to include",
    ),
    filt_license: bool = typer.Option(
        True,
        help="Remove non-permissive licenses",
    ),
    replace_non_permissive_cols: list[str] = typer.Option(
        [],
        "--replace-non-permissive-col",
        help="Replace non-permissively licensed col with None",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Test/Dry run with 500 sample slice of data",
    ),
    max_threads: int = typer.Option(
        8,
        "--max-threads",
        help="Max threads to use, e.g. polars env variable",
    ),
    clear_temp: bool = typer.Option(
        True,
        "--clear-temp",
        help="Delete temp files, whether successful or not",
    ),
    runtime_name: str | None = typer.Option(
        None,
        "--runtime",
        "-r",
        help="Execution backend (local or modal; defaults to [runtime].default)",
    ),
    source_backend: str | None = source_backend_arg(),
    source_opts: list[str] = source_opt_arg(),
    source_base_dir: Path | None = source_base_dir_arg(),
    source_volume: str | None = source_volume_arg(),
    tracking_uri: str | None = typer.Option(
        None,
        "--tracking-uri",
        help="Override MLflow tracking URI",
    ),
    artifact_loc: Path | None = typer.Option(
        None,
        "--artifact-loc",
        help="Override artifact storage location",
    ),
) -> None:
    """Run the preprocessing pipeline on a dataset.

    Worked example:

        forecite preprocess <origin> --op "clean:title" --lowercase \
            --trim-min-chars 20 --op "dropna:title" --op "embed:text,abstract" \
            --op-embedder modernbert-base

    Op specs are "<kind>:<colspec>" where colspec is a comma-separated list
    of columns. Per-op params apply to clean ops in order; --op-tokeniser and
    --op-embedder(-config) attach to the most recent tokenise/embed op.
    """
    if bool(rows_per_part) and bool(n_partitions):
        raise typer.BadParameter(
            "Define partition split with one or the other "
            "(--partitions or --rows-per-part), not both"
        )

    env = load_env(
        overrides={
            k: v
            for k, v in {
                "tracking_uri": tracking_uri,
                "artifact_loc": artifact_loc,
            }.items()
            if v is not None
        }
    )

    source_backend_obj = build_source_backend_from_cli(
        env=env,
        source_backend=source_backend,
        source_opts=source_opts,
        source_base_dir=source_base_dir,
        source_volume=source_volume,
    )

    if runtime_name is None:
        runtime_name = env.runtime.get("default", "local")

    if runtime_name == "modal":
        if "modal" not in env.runtime:
            raise typer.BadParameter(
                "--runtime modal requires a [runtime.modal] section in "
                "config/config.toml. See config/config.example.toml."
            )

    runtime = build_runtime(runtime_name, env)

    supported = getattr(runtime, "supported_source_backends", {"local"})
    if source_backend_obj.name not in supported:
        raise typer.BadParameter(
            f"Runtime {runtime_name!r} does not support source backend "
            f"{source_backend_obj.name!r}"
        )

    origin_source, base_name = _resolve_origin(origin, source_backend_obj)

    dest_name = name if name else f"{base_name}-preprocessed"
    destination = source_backend_obj.get_source(dest_name)

    steps = _parse_ops(
        ops,
        lowercase=lowercase,
        trim_min_chars=trim_min_chars,
        trim_max_sigma=trim_max_sigma,
        require_terminal_period=require_terminal_period,
        lang_policy=lang_policy,
        drop_quality=[not v for v in no_drop_quality],
        tokeniser=tokeniser,
        embedder_key=embedder_key,
        embedder_kwargs_json=embedder_kwargs_json,
    )

    job = PreprocessJob(
        origin=origin_source,
        destination=destination,
        steps=steps,
        n_partitions=n_partitions,
        rows_per_part=rows_per_part,
        compression_level=compression_level,
        start_date=start_date,
        end_date=end_date,
        field_id=field_id,
        languages=languages,
        types=types,
        filt_license=filt_license,
        replace_non_permissive_cols=replace_non_permissive_cols,
        dry_run=dry_run,
        max_threads=max_threads,
        runtime_name=runtime_name,
        params=dict(ctx.params),
    )

    result = runtime.run_preprocess(job)
    logger.info(f"Preprocessed dataset written to {result.resolve()}")


if __name__ == "__main__":
    app()
