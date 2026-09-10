# Roadmap

Full version history and forward plans for text-forecast. The short summary
lives in the [README](README.md#2-road-map); the authoritative plan status
table lives in [`plans/releases/README.md`](plans/releases/README.md).

Structure: each major version is a top-level collapsible section; the patch
releases inside it are nested collapsible sections. Dates for the v0.x–v2.2
era are not recorded anywhere in the repo; later versions carry the status
dates given in their release plans.

<details>
<summary><b>v0.2 — Data pre-processing CLI</b></summary>

> <details>
> <summary><b>v0.2.0 Data pre-processing CLI</b></summary>
>
> - Standardise data pre-processing notebook into separate functions.
> - Orchestrate pre-processing functions in main loop.
> - CLI control via Typer app, functions induced/skipped via option flags.
>
> </details>

</details>

<details>
<summary><b>v0.3 — Dataloader flexibility</b></summary>

> <details>
> <summary><b>v0.3.0 Dataloader Flexibility</b></summary>
>
> - Support concatenation of multiple string/token columns when serving examples from dataloader.
>
> </details>

</details>

<details>
<summary><b>v0.4 — Evaluation CLI &amp; metric tracker</b></summary>

> <details>
> <summary><b>v0.4.0 Eval CLI</b></summary>
>
> - CLI app for evaluating models
> - Config driven with optional CLI overrides
> - Temporal scrub with variable intervals and MLflow tracking
> - JSON data export for website historical performance graphs
>
> </details>
>
> <details>
> <summary><b>v0.4.1 Metric Tracker Generalisation</b></summary>
>
> - Generalise & simplify metric tracker methods/initialisation
> - Generalised parameter init function & metric calc functions responsive to store name prefix
>
> </details>
>
> <details>
> <summary><b>v0.4.2 Standardise cross-app arg parsing</b></summary>
>
> - Standardise start/end date arg parsing across apps
>
> </details>
>
> <details>
> <summary><b>v0.4.3 Update data loading & handling</b></summary>
>
> - Specialised dataset load funcs, more efficient mem use
> - Ignore extra cols (e.g index) during parquet data load
> - Add no. of examples in dataset to train/eval metric/param logging
>
> </details>
>
> <details>
> <summary><b>v0.4.4 Enhanced Metric calculation</b></summary>
>
> - Add 'best threshold' metric calculation for accuracy, also recording precicion/recall at that threshold
> - PR & ROC AUC charts logged as mlflow artifacts during evaluation step
>
> </details>
>
> <details>
> <summary><b>v0.4.5 Preprocess &amp; Train app upgrade</b></summary>
>
> - Add metadata export, drop count logging, more user warnings, to pre-processing
> - Cleaning has more granular control with column specific 'levels', tidied argument names
> - Cleaning split into 'drop' and 'clear' (replace with null)
> - License based filtering and clearing added to clean
> - Tokenise step leaves empty list for nulls instead of filling nulls with empty string prior to tokenisation
> - Pre-processing sped up by partitioning via filter instead of slicing
> - Log model file as mlflow arifact
>
> </details>

</details>

<details>
<summary><b>v0.5 — Dataset caching, descriptives, and efficiency</b></summary>

> <details>
> <summary><b>v0.5.0 Dataset ipc refactor</b></summary>
>
> - Moved from in mem df to ipc cached dataset post filtering operations for fast random access of OOM dataset rows
> - Now organise dataset & their cache under the 'name' attribute
>
> </details>
>
> <details>
> <summary><b>v0.5.1 Descriptives app</b></summary>
>
> - Added descriptives app, allowing CLI driven descriptives generation of specific columns across datasets
> - Shows Polars descriptives table and relative/total frequency counts for variable bucket boundaries
> - Calculates proportional weights for balanced training with n_buckets classes
>
> </details>
>
> <details>
> <summary><b>v0.5.2 Metric tracker efficiency &amp; Model train checkpoint loading</b></summary>
>
> - Removed redundant recall/precision calculations from best accuracy metric calc
> - Added model checkpoint to be loaded in train loop
>
> </details>
>
> <details>
> <summary><b>v0.5.3 Efficiency, Control &amp; Clean update</b></summary>
>
> - More efficient best accuracy calculations across a smaller range & no. of values
> - Train loop can load mlflow stored checkpoints & set parent run via CLI args
> - Dataset formatting classes available via registry in CLI / config
> - Dataset kwargs specified in CLI / config for flexible dataset initialisation
> - Polars max threads, and compile mode determined by CLI args
> - Learning rate scheduler args determined by config
> - New lowercase clean method for string columns
>
> </details>

</details>

<details>
<summary><b>v0.6 — Metric export &amp; visualisation</b></summary>

> <details>
> <summary><b>v0.6.0 Model Eval Metric export for visualisation</b></summary>
>
> - Dataset outputs dataclass for resilient batch access to optional fields with dot operator access
> - Associate model outputs with input row id to categorise and measure correlation of metrics
> - Structured JSON metric exports for modularised records
>
> </details>

</details>

<details>
<summary><b>v0.7 — CLI &amp; config consolidation</b></summary>

> <details>
> <summary><b>v0.7.0 CLI &amp; Config Consolidation</b></summary>
>
> - Add config/env value overrides to train app
> - Move code-as-config module from root to src, allow config value overrides from CLI via option flags
> - Refactor train (main) loop into /apps
> - Create metric tracker base class for tracking logic, overide metric calculation in children
> - Seperate train/data/env configs into distinct files
> - Add dedicated loss/optimisation config for lr schedule milestones etc.
>
> </details>

</details>

<details>
<summary><b>v1.0 — Dependency-injected experiment configs</b></summary>

> <details>
> <summary><b>v1.0.0 Dependency-injected experiment configs</b></summary>
>
> - One self-contained experiment module per experiment under `src/config/experiments/` declares the full object graph
> - Generic `Experiment[T_Batch]` dataclass holds model, strategy, tracker, plain torch DataLoaders, and checkpoint processor
> - `Engine` owns the epoch/batch loop; `Strategy` implements `training_step`, `validation_step`, and `configure_optimizers`
> - Typed optimizer/scheduler specs (`AdamWSpec`, `WarmupCosineSpec`) built inside the experiment file
> - Checkpoint processor abstraction (local, MLflow, S3 stub) saves full dict checkpoints and the original experiment file as a run artifact
> - `train` and `eval` apps rewired to load experiments; `eval` downloads the run's experiment file and rebuilds its windows
> - Tracker rewrite: dedicated subclasses, dict CPU stores, explicit kwargs, no singleton config reads
> - Runtime `Registry` replaced by `@component` marker + `utils.build_helper` for package `__init__` blocks
> - basedpyright strict, Python 3.13, keyword-only constructors throughout
>
> </details>

</details>

<details>
<summary><b>v1.1 — Runtime abstractions &amp; robust preprocessing</b></summary>

> <details>
> <summary><b>v1.1.0 Migrate remaining apps + robust preprocessing + local/Modal prep</b></summary>
>
> - Migrate `preprocess`, `describe`, and `engineer` apps to load machine settings via `config.env.load_env(...)`
> - Add `[env]` CLI override flags to all remaining apps
> - Remove the temporary Phase-0 `config/env.py` `__getattr__` shim
> - Make preprocessing robust: lazy embedder loading, pluggable embedding models, CPU/CUDA support
> - Introduce `Runtime` and `DataSource` abstractions so local and Modal execution share the same app code
>
> </details>

</details>

<details>
<summary><b>v1.2 — Local / Modal runtime split</b></summary>

> <details>
> <summary><b>v1.2.0 Local / Modal runtime split</b></summary>
>
> - Implement `modal` runtime backend using the abstractions from v1.1
> - Add Modal volume-backed data sources and Modal GPU embedder
> - Run `preprocess` with `--runtime modal` (train/eval stay local; see v2.0)
>
> </details>

</details>

<details>
<summary><b>v1.3 — Modal describe &amp; engineer</b></summary>

> <details>
> <summary><b>v1.3.0 Modal runtime for describe and engineer</b></summary>
>
> - Runtime-agnostic `DescribeJob` / `EngineerJob` pipelines under `src/data/pipeline/`, moved out of the apps
> - `Runtime` protocol extended with `get_source`, `run_describe`, `run_engineer`; local and Modal backends implement it
> - `describe` and `engineer` dispatch through the runtime with `--runtime modal`; no app-side runtime branching
> - Describe and engineer run as whole CPU jobs in the shared `text-forecast-data` Modal app against the staged volume
> - Engineer writes `metadata.json` recording the runtime that produced the dataset
>
> </details>

</details>

<details>
<summary><b>v2.0 — Training &amp; evaluation on Modal</b></summary>

> <details>
> <summary><b>v2.0.0 Training and evaluation on Modal</b></summary>
>
> - Experiments build from `(runtime, env)` alone; machine locations live in `[source]` config, never CLI flags
> - Venue-independent `TrainJob` / `EvalJob` pipelines under `src/training/pipeline/`; apps are thin parse-and-delegate layers
> - MLflow runs created client-side by the runtime wrapper and resumed in-place by `run_id`, so run naming/parenting works the same on both venues
> - `ModalTrainingGPU` spawns fire-and-forget train/eval jobs with `.spawn()`; checkpoints land as MLflow artifacts, so a Modal-trained run resumes locally and vice versa
> - Eval prediction exports are `exports/<year>` MLflow artifacts (downloadable from anywhere), not local directories
> - Engine and trackers run headless (`progress=None`) inside containers
>
> </details>

</details>

<details>
<summary><b>v2.1 — Clean pipeline refactor</b></summary>

> <details>
> <summary><b>v2.1.0 Ordered step specs, per-column intensity controls, nested CLI</b></summary>
>
> - Replace the integer clean `level` with per-operation step specs
> - Preprocessing steps become an ordered list on the job instead of a fixed call order in `run_preprocess_pipeline`
> - Preprocess CLI surface rebuilt on top of the step-spec model
> - Regression test for the suspected length-filter sign error in `src/data/preprocess/clean.py`
> - Implemented (T0–T5); one manual `--runtime modal` dry run still pending
>
> </details>

</details>

<details>
<summary><b>v2.3 — Retrieval-forecast model &amp; vector store</b></summary>

> <details>
> <summary><b>v2.3.0 Retrieval-forecast model, vector store dataset, experiment config</b></summary>
>
> - Vector store dataset: precomputed parquet embedding columns loaded into an in-memory (FAISS or matmul) index
> - `RetrievalForecast` model with differentiable straight-through top-k retrieval over a candidate pool
> - `retrieval_forecast` experiment DI config wiring dataset and model together
> - Additive only; no existing CLI surface, dataset, or pipeline behaviour changes
> - Tagged `v2.3.0`
>
> </details>

</details>

<details>
<summary><b>v2.4 — Expanded query search</b></summary>

> <details>
> <summary><b>v2.4.0 / v2.4.1 — Expanded query search &amp; bf16 metric fix</b></summary>
>
> - `candidate_strategy` on `RetrievalModelConfig`: `nearest` (default), `random`, `mixed` stage-1 pool selection
> - Independent corpus scope: `t_start`/`t_end` bounds on `VectorStoreDatasetConfig` and a separate corpus filter expression in the experiment file
> - Compute dtype plumbing: `--dtype fp32|bf16|fp16` on the train CLI, flowing through `TrainJob` into `RunContext.dtype`; retriever buffers and stage-2 pool gather in the compute dtype
> - v2.4.1: bfloat16 metric stores convert to fp32 at the `_gather_store` boundary, covering all five tracker subclasses and the export path; regression test added
> - Carries the fixes planned as v2.3.1 (MLflow logging and param-tracking: train_ids feed, latched empty-store warning, eval tracking-URI order, logger.exception swaps); v2.3.1 never shipped as its own version number
> - Tagged `v2.4.1` (no `v2.4.0` tag exists)
>
> </details>
>
> <details>
> <summary><b>v2.4.2 — Id-feed and tracking-store fixes</b></summary>
>
> - `ClassificationStrategy` feeds `train_ids` symmetrically with `val_ids` (NaN-guarded)
> - Empty-store warning latches: one WARNING per store name per tracker instead of per batch
> - The synchronous per-batch MLflow logging item was struck (hang reproduced with it)
> - Tagged `v2.4.2`
>
> </details>
>
> <details>
> <summary><b>v2.4.3 — Candidate-strategy Gumbel sampling</b></summary>
>
> - `VectorRetriever.search` draws the seeded Gumbel noise on-device (random/mixed search speed)
> - Backend-only `GUMBEL_TAU` smoothing constant in `src/models/retrieval_forecast.py`; not config-exposed
> - Tagged `v2.4.3`
>
> </details>
>
> <details>
> <summary><b>v2.4.4 — Preprocess embed/tokenise execution restored</b></summary>
>
> - `run_preprocess_pipeline` executes `EmbedStep` and `TokeniseStep` handlers again (the step loop's `isinstance` guard silently skipped them)
> - Multi-column embed ops fixed: bos/eos literal wrapping aliased back to the source column (polars duplicate-`literal` crash)
> - Stray `import torch` removed from `src/data/preprocess/pipeline.py`
> - Prior `embed:`/`tokenise:` datasets require regeneration
> - Tagged `v2.4.4`
>
> </details>

</details>

## Planned

Forward-looking plans (2.2 CLI launch speed, 2.5.0 Modal runtime tidy-up,
2.6 production hardening, 2.7 distributed training, 2.8 remote progress)
live in [`plans/releases/README.md`](plans/releases/README.md), which is the
authoritative shipped-vs-pending status table.
