# CLI apps

`src/apps/` holds the Typer applications mounted by `src/main.py`. Each app is a
thin layer: it parses flags, resolves the environment, builds a job object,
and delegates to `runtime.run_train(...)` / `runtime.run_eval(...)`. The app
never branches on the runtime; the `Runtime` implementation owns execution.

All apps are invoked as subcommands of `python -m src.main`, e.g.
`python -m src.main train --name my-run --experiment transformer_class`.
The root command resolves the experiment name from `--experiment` or
`config/config.toml`.

## Shared options (`source_args.py`)

The data-pipeline apps (preprocess, describe, engineer) accept source/env flags:

- `--source-backend` - `local` (default) or `modal`; defaults to `[source].default`
- `--source-opt key=value` - backend option overrides, repeatable; dotted keys nest
- `--source-base-dir` / `--source-volume` - shortcuts for local/modal backends
- `--tracking-uri`, `--artifact-loc` - override MLflow settings from env

`train` and `eval` do not take source flags. Experiments resolve their dataset
location from `env.source` (`[source].default` plus its config section), so
dataset identity lives in the experiment file and location in the environment.

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
`<col>_len`) and years-to-first-citation (on by default; needs date-typed
`publication_date` and list-typed `referenced_works`/`counts_by_year_years`).
Input comes from `--input-path` or `--input-dataset`; output defaults to
`<input>-engineered`, overridable with `--output` (path or dataset name);
`--n-partitions` sets the partition split. Supports `--runtime modal`. Writes
`metadata.json` with the job parameters and runtime. `--dry-run` slices 50 rows.

## eval

Evaluates a trained run over sliding time windows. Requires an MLflow `--run-id`
and `--epoch` to load the checkpoint from, plus a window spec: `--start-date`,
`--interval` (in years; the only unit currently supported) and optional
`--end-date`.

The app builds an `EvalJob`; `LocalRuntime.run_eval` creates the eval run
client-side in `<experiment>-EVAL`, then runs `run_eval_pipeline` locally. For
each window it slices the validation `GraphDataset` with `with_window`, runs
one eval epoch through the Engine, and logs both metrics and prediction exports
as MLflow artifacts (`exports/<year>`) on the eval run. The experiment file is
downloaded from the checkpoint store, so the exact training config is reproduced
without CLI re-specification. Only `GraphDataset`-based experiments are
currently supported. `--no-gpu` forces CPU; `--clean-up` deletes temp files
afterwards.

With `--runtime modal`, `ModalRuntime.run_eval` creates the eval run
client-side (same `<experiment>-EVAL` naming), maps the job onto the Modal
volumes (staged dataset + checkpoint scratch), and spawns a `ModalTrainingGPU`
container that runs the identical pipeline headless. It returns immediately
with the eval run id and the Modal FunctionCall id; follow the job with
`modal.FunctionCall.from_id(<id>).get()`. Exports land as
`exports/<year>` artifacts on that run either way.

## train

Main training entry point. The app reads the experiment file's bytes into a
`TrainJob`; `LocalRuntime.run_train` creates the MLflow run client-side, then
`run_train_pipeline` loads the experiment from those bytes, optionally
torch-compiles the model (`--compile mode`, `--fullgraph`), restores state from
`--load-id` + `--load-epoch` if resuming (`--model-only` skips
optimizer/scheduler), and runs `engine.fit()` inside the pre-created run.

Run naming: `--name` sets it explicitly, or `--suffix` produces
`<ModelClass>-<suffix>`; exactly one of the two is required. `-DRY` is appended
when `--subsample` is set. The final name is applied once the model class is
known inside the pipeline. The experiment module file and model source file are
logged as artifacts, along with dataset sizes and model class. Checkpoints go
through the experiment's configured processor (local, MLflow store, or S3).

`--runtime` to select the execution backend (overrides `[runtime].default`).

With `--runtime modal`, `ModalRuntime.run_train` creates the MLflow run
client-side (plan P6), maps the job onto the Modal volumes, and spawns a
`ModalTrainingGPU` container running the identical pipeline headless. It
returns immediately with the MLflow run id and the Modal FunctionCall id
(fire-and-forget, plan P7); poll with
`modal.FunctionCall.from_id(<id>).get()` or open the run in the MLflow UI.
Checkpoints written in the container are uploaded as MLflow artifacts, so a
Modal-trained run resumes locally (and vice versa) with no extra steps.
`torch.compile` is dropped when GPU snapshotting is enabled, since
compilation breaks snapshot restore; this is a dispatch decision and
experiment files are untouched.

### Modal config

`[runtime.modal]` in `config/config.toml` holds every Modal setting. The data
apps use `project`, `gpu`, `python_version`, and `embedder_batch_size`. Train
and eval additionally use `train_gpu`, `timeout`, `raw_volume`,
`staged_volume`, `checkpoint_volume`, and the optional `tracking_uri` override
(empty means inherit `[env].tracking_uri`).
`config.env.modal_runtime_config(env)` is the typed accessor; pass
`require_checkpoint_volume=True` on train/eval dispatch paths so a missing
`checkpoint_volume` fails fast with the key to set.

Useful flags: `--parent-id` to nest MLflow runs, `--start-epoch` to override the
resume epoch, `--progress/--no-progress` for the Rich bars, `--gpu/--no-gpu`,
`--runtime` to select the execution backend (overrides `[runtime].default`).
