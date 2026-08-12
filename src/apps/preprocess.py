from __future__ import annotations

import json
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
from config.env import Env, load_env
from data.preprocess.pipeline import PreprocessJob
from data.sources import LocalSourceBackend, SourceBackend
from data.sources.base import DataSource
from runtime import build_runtime
from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

app = typer.Typer(pretty_exceptions_enable=False)


def _parse_embedder_config(value: str) -> dict[str, object]:
    if not value:
        return {}
    return json.loads(value)


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


@app.callback(invoke_without_command=True)
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
    clean_cols: list[str] = typer.Option(
        [],
        "--clean-col",
        "-c",
        help="Data columns to clean, must be str; pass multiple flags for multiple cols",
    ),
    clean_levels: list[int] = typer.Option(
        [],
        "--clean-level",
        "-cl",
        help="Level to clean each column at; must be in the same order as --clean-col",
    ),
    clean_min_len: list[int] = typer.Option(
        [],
        "--clean-min-len",
        "-cml",
        help="Minimum length for the column being cleaned; same order as --clean-col",
    ),
    tokeniser: str = typer.Option(
        "",
        "--tokeniser",
        "-t",
        help="Tokenizer to use when tokenise step is requested",
    ),
    tokenise_cols: list[str] = typer.Option(
        [],
        "--tokenise-col",
        "-tc",
        help="Data columns to tokenise; pass multiple flags for multiple cols",
    ),
    embedder: str = typer.Option(
        "",
        "--embedder",
        "-em",
        help="Registered embedder key; inactive if empty",
    ),
    embedder_config: str = typer.Option(
        "",
        "--embedder-config",
        help="JSON object passed to the embedder constructor",
    ),
    embedder_batch_size: int = typer.Option(
        32,
        "--embedder-batch-size",
        help="Batch size for the embedder",
    ),
    embedder_device: str = typer.Option(
        "",
        "--embedder-device",
        help="Device override for the embedder (e.g. cpu, cuda)",
    ),
    embed_cols: list[str] = typer.Option(
        [],
        "--embed-cols",
        "-ec",
        help="Data columns to embed; pass multiple flags for multiple cols",
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
    drop_na_cols: list[str] = typer.Option(
        [],
        "--drop-na-col",
        help="Cols to drop nulls in",
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
    assert (tokeniser and tokenise_cols) or (not tokeniser and not tokenise_cols), (
        "Provide columns to be tokenised"
    )
    assert (embedder and embed_cols) or (not embedder and not embed_cols), (
        "Provide columns to be embedded"
    )
    assert len(clean_cols) == len(clean_levels) == len(clean_min_len), (
        "Clean cols, levels, and min-lens must be given in same quantity"
    )
    assert not (rows_per_part and n_partitions), (
        "Define partition split with one or the other"
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
        if embedder_device:
            logger.warning(
                "--embedder-device is ignored when using --runtime modal; "
                "the Modal GPU class always uses cuda."
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

    embedder_kwargs = _parse_embedder_config(embedder_config)
    if embedder_device:
        embedder_kwargs["device"] = embedder_device
    if embedder_batch_size:
        embedder_kwargs["batch_size"] = embedder_batch_size

    job = PreprocessJob(
        origin=origin_source,
        destination=destination,
        clean_cols=clean_cols,
        clean_levels=clean_levels,
        clean_min_len=clean_min_len,
        tokeniser=tokeniser,
        tokenise_cols=tokenise_cols,
        embedder_key=embedder,
        embedder_kwargs=embedder_kwargs,
        embed_cols=embed_cols,
        n_partitions=n_partitions,
        rows_per_part=rows_per_part,
        compression_level=compression_level,
        start_date=start_date,
        end_date=end_date,
        field_id=field_id,
        drop_na_cols=drop_na_cols,
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
