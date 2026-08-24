# CLI apps

`src/apps/` holds the Typer applications mounted by `src/main.py`. Each app is a
thin layer: it parses flags, resolves the environment and data source, builds a
job object or experiment, then hands execution to a runtime or the Engine.

All apps are invoked as subcommands of `python -m src.main`, e.g.
`python -m src.main train --name my-run --experiment transformer_class`.
The root command resolves the experiment name from `--experiment` or
`config/config.toml`.

## Shared options (`source_args.py`)

Every app accepts the same source/env flags:

- `--source-backend` - `local` (default) or `modal`; defaults to `[source].default`
- `--source-opt key=value` - backend option overrides, repeatable; dotted keys nest
- `--source-base-dir` / `--source-volume` - shortcuts for local/modal backends
- `--tracking-uri`, `--artifact-loc` - override MLflow settings from env

## preprocess

Runs the cleaning/tokenising/embedding pipeline. Takes an origin dataset as an
argument (dataset name or path under the source base dir) and writes
`<origin>-preprocessed` unless `--name` is given.

Steps are opt-in per column:

- `--clean-col` + `--clean-level` + `--clean-min-len` - text cleaning; the three
  lists must be the same length
- `--tokeniser` + `--tokenise-col`
- `--embedder` + `--embed-col`, with `--embedder-config` (JSON), batch size, device

Also filters rows by date, field id, language, document type, and license
(non-permissive licenses are dropped unless overridden). Output is partitioned
via `--partitions` or `--rows-per-part` with zstd compression. Supports
`--runtime modal` for GPU embedding; `--dry-run` slices 500 rows.

## describe

Dataset profiling. Prints statistics over columns chosen with `--column`,
optionally bucketed by value ranges (`--bucket`) and filtered by time window
(`--start-time`/`--end-time` on `--time-col`) or a raw Polars expression
(`--filter-expr`). Runs on any runtime.

## engineer

Feature engineering pass. Computes column lengths (`--calc-len`, written as
`<col>_len`) and years-to-first-citation (on by default). Input comes from
`--input-path` or `--input-dataset`; output defaults to `<input>-engineered`,
overridable with `--output` (path or dataset name). `--dry-run` uses a 500-row
slice.

## eval

Evaluates a trained run over sliding time windows. Requires an MLflow `--run-id`
and `--epoch` to load the checkpoint from, plus a window spec: `--start-date`,
`--interval` (in years; the only unit currently supported) and optional
`--end-date`.

For each window it slices the validation `GraphDataset` with `with_window`,
runs one eval epoch through the Engine, exports predictions to
`<base_dir>/eval/<run_id>/run-<prefix>/<year>`, and logs metrics to MLflow under
`<experiment>-EVAL`. The experiment file is downloaded from the checkpoint
store, so the exact training config is reproduced without CLI re-specification.
Local-only: requires `--source-backend local`. `--no-gpu` forces CPU;
`--clean-up` deletes temp checkpoints afterwards.

## train

Main training entry point. Loads the selected experiment module via
`config.loader.load_experiment`, optionally torch-compiles the model
(`--compile mode`, `--fullgraph`), restores state from `--load-id` +
`--load-epoch` if resuming (`--model-only` skips optimizer/scheduler), then runs
`engine.fit()` inside an MLflow run.

Run naming: `--name` sets it explicitly, or `--suffix` produces
`<ModelClass>-<suffix>`; exactly one of the two is required. `-DRY` is appended
when `--subsample` is set. The experiment module file and model source file are
logged as artifacts, along with dataset sizes and model class. Checkpoints go
through the experiment's configured processor (local, MLflow store, or S3).

Useful flags: `--parent-id` to nest MLflow runs, `--start-epoch` to override the
resume epoch, `--progress/--no-progress` for the Rich bars, `--gpu/--no-gpu`.
