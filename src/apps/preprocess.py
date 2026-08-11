from __future__ import annotations

import json
import math
import os
from datetime import datetime
from logging import getLogger
from pathlib import Path

import polars as pl
import typer
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from config.env import Env, load_env
from data.preprocess import clean_step, tokenise_step
from data.preprocess.embed import TextEmbedder
from runtime import LocalRuntime, Runtime
from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

app = typer.Typer(pretty_exceptions_enable=False)


def _parse_embedder_config(value: str) -> dict[str, object]:
    if not value:
        return {}
    return json.loads(value)


def _make_runtime(runtime_name: str, env: Env) -> Runtime:
    del env
    if runtime_name == "local":
        return LocalRuntime()
    raise typer.BadParameter(
        f"Runtime {runtime_name!r} is not supported in v1.1. "
        "Modal runtime is planned for v1.2."
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    origin: str = typer.Argument(
        help="Path to raw data or dataset name under raw-loc",
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
    runtime_name: str = typer.Option(
        "local",
        "--runtime",
        "-r",
        help="Execution backend (only 'local' is supported in v1.1)",
    ),
    tracking_uri: str | None = typer.Option(
        None,
        "--tracking-uri",
        help="Override MLflow tracking URI",
    ),
    raw_loc: Path | None = typer.Option(
        None,
        "--raw-loc",
        help="Override raw data location",
    ),
    staged_loc: Path | None = typer.Option(
        None,
        "--staged-loc",
        help="Override staged data location",
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
                "raw_loc": raw_loc,
                "staged_loc": staged_loc,
                "artifact_loc": artifact_loc,
            }.items()
            if v is not None
        }
    )
    runtime = _make_runtime(runtime_name, env)

    os.environ["POLARS_MAX_THREADS"] = f"{max_threads}"

    def measure_lf(lf: pl.LazyFrame, run: bool = False) -> float:
        if run:
            return lf.select(pl.len()).collect(engine="streaming").item()
        return float("nan")

    origin_path = Path(origin)
    if origin_path.exists():
        source_path = origin_path
        base_name = origin_path.stem
    else:
        source = runtime.get_source(root=env.raw_loc, name=origin)
        source_path = runtime.maybe_download(source)
        base_name = origin

    if not name:
        destination = runtime.get_source(
            root=env.staged_loc, name=f"{base_name}-preprocessed"
        )
    else:
        destination = runtime.get_source(root=env.staged_loc, name=name)

    destination_path = destination.resolve()
    os.makedirs(destination_path, exist_ok=True)
    logger.info(f"Preprocessing {source_path} -> {destination_path}")

    lf_whole: pl.LazyFrame = pl.scan_parquet(
        list(source_path.glob("*.par*")), extra_columns="ignore"
    )

    if filt_license:
        lf_whole = lf_whole.filter(pl.col("is_license_safe"))
    else:
        logger.warning("Including non-permissive licenses")

    if start_date is not None:
        lf_whole = lf_whole.filter(pl.col("publication_date") >= start_date)
    else:
        logger.warning("No start date set")

    if end_date is not None:
        lf_whole = lf_whole.filter(pl.col("publication_date") < end_date)
    else:
        logger.warning("No end date set")

    if field_id:
        lf_whole = lf_whole.filter(pl.col("field_id").is_in(field_id))
    else:
        logger.warning("No field id filter set")

    if languages:
        lf_whole = lf_whole.filter(pl.col("language").is_in(languages))
    else:
        logger.warning("No Language filters set")

    if types:
        lf_whole = lf_whole.filter(pl.col("type").is_in(types))
    else:
        logger.warning("No document type filters set")

    n_rows: float = measure_lf(lf_whole, True)
    if n_partitions:
        rows_per_part = math.ceil(n_rows / n_partitions)
    elif rows_per_part:
        n_partitions = math.ceil(n_rows / rows_per_part)

    lf_whole = lf_whole.with_row_index("idx").with_columns(
        (pl.col("idx") // rows_per_part).clip(0, n_partitions - 1).alias("part")
    )

    progress_bar = Progress(
        TextColumn("[bold blue] {task.description}", justify="left"),
        BarColumn(bar_width=40),
        TextColumn("[task.completed]{task.completed}/{task.total}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        speed_estimate_period=60.0 * 10,
    )
    progress_bar.start()
    progress = progress_bar.add_task("Rows", total=int(n_rows))

    if dry_run:
        lf_whole = lf_whole.slice(0, 500)

    embedder_instance: TextEmbedder | None = None
    if embedder:
        config = _parse_embedder_config(embedder_config)
        if embedder_device:
            config["device"] = embedder_device
        if embedder_batch_size:
            config["batch_size"] = embedder_batch_size
        embedder_instance = runtime.get_embedder(key=embedder, **config)

    def embed_batch(series: pl.Series) -> pl.Series:
        assert embedder_instance is not None
        texts = [t if isinstance(t, str) else "" for t in series.to_list()]
        vectors = embedder_instance.encode(texts)
        return pl.Series(
            vectors,
            dtype=pl.Array(pl.Float32, width=embedder_instance.output_dim),
        )

    for i in range(n_partitions):
        lf = lf_whole.filter(pl.col("part") == i).drop(["idx", "part"])

        for col in replace_non_permissive_cols:
            logger.info(f"Replacing non-permissive {col} with None")
            lf = lf.with_columns(
                pl.when(pl.col("is_license_safe") == False)
                .then(pl.lit(None, dtype=lf.schema[col]))
                .otherwise(pl.col(col))
                .alias(col)
            )

        if clean_cols:
            for col, level, min_len in zip(clean_cols, clean_levels, clean_min_len):
                l1 = measure_lf(lf)
                lf = clean_step(
                    lf=lf,
                    col=col,
                    min_len=min_len,
                    level=level,
                )
                l2 = measure_lf(lf)
                logger.info(f"Dropped {l1 - l2:,.0f} in {col} at clean lvl {level}")
        elif i == 0:
            logger.warning("No cols selected for cleaning")

        if drop_na_cols:
            for col in drop_na_cols:
                lf = lf.drop_nulls(subset=col)
        elif i == 0:
            logger.warning("No columns set to drop nulls")

        if embedder_instance:
            if i == 0:
                logger.info(f"Embedding {', '.join(embed_cols)} with {embedder}")
            lf = (
                lf.drop_nulls(embed_cols)
                .with_columns(
                    to_embed=pl.concat_str(
                        [pl.col(col) for col in embed_cols],
                        separator=" ",
                    )
                )
                .with_columns(
                    pl.col("to_embed")
                    .map_batches(
                        embed_batch,
                        return_dtype=pl.Array(
                            pl.Float32, width=embedder_instance.output_dim
                        ),
                    )
                    .alias(f"{' '.join(embed_cols)}_embedding")
                )
                .drop("to_embed")
            )

        if tokeniser:
            if i == 0:
                logger.info(f"Tokenising {', '.join(tokenise_cols)} with {tokeniser}")
            lf = tokenise_step(
                lf=lf,
                tokeniser_path=tokeniser,
                columns=tokenise_cols,
            )

        if dry_run:
            print(lf.collect())
            break

        output_fname = destination_path / f"part_{i}.parquet"
        lf.sink_parquet(
            output_fname,
            statistics=True,
            compression="zstd",
            compression_level=compression_level,
        )
        n_written = (
            pl.scan_parquet(output_fname)
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )
        progress_bar.update(progress, advance=n_written)

    metadata = {k: str(v) for k, v in ctx.params.items()}
    metadata["env"] = {
        "tracking_uri": env.tracking_uri,
        "raw_loc": str(env.raw_loc),
        "staged_loc": str(env.staged_loc),
        "artifact_loc": str(env.artifact_loc),
    }
    with open(destination_path / "metadata.json", "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    app()
