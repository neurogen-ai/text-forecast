# Changelog

Completed releases for text-forecast, newest first. Structured like
[ROADMAP.md](ROADMAP.md): majors are top-level collapsible sections, patches
are nested inside. Only shipped versions appear here — forward plans live in
[`plans/releases/README.md`](plans/releases/README.md).

Dates: the v0.x–v2.2 era is not dated anywhere in the repo (date unknown).
Later entries carry the implementation dates recorded in their release
plans; the git tags `v2.3.0, v2.4.1, v2.4.2, v2.4.3, v2.4.4` corroborate
v2.3+ only.

<details>
<summary><b>v2.x — Hybrid local/Modal cloud system</b></summary>

> <details>
> <summary><b>v2.4 — Expanded query search (2026-09-06 → 2026-09-09)</b></summary>
>
> > <details>
> > <summary><b>v2.4.4 — Preprocess embed/tokenise execution restored</b></summary>
> >
> > - `EmbedStep`/`TokeniseStep` handlers execute again (previously parsed then silently skipped)
> > - Multi-column embed ops: bos/eos literal wrapping aliased back to the source column (polars duplicate-`literal` crash fixed)
> > - Stray `import torch` removed from the preprocess pipeline; prior `embed:`/`tokenise:` datasets need regeneration
> >
> > </details>
> >
> > <details>
> > <summary><b>v2.4.3 — Candidate-strategy Gumbel sampling</b></summary>
> >
> > - Seeded Gumbel noise drawn on-device for random/mixed candidate pools (search speed)
> > - `GUMBEL_TAU` smoothing constant is backend-only, not config-exposed
> >
> > </details>
> >
> > <details>
> > <summary><b>v2.4.2 — Id-feed and tracking-store fixes</b></summary>
> >
> > - `train_ids` fed symmetrically with `val_ids` (NaN-guarded)
> > - Empty-store warning latches per store name per tracker
> > - Synchronous per-batch MLflow logging struck from scope (hang reproduced with it)
> >
> > </details>
> >
> > <details>
> > <summary><b>v2.4.1 — bf16 metric-store fix (with v2.4.0)</b></summary>
> >
> > - bfloat16 metric stores convert to fp32 at the `_gather_store` boundary (all tracker subclasses + export path); regression test added
> > - Ships the v2.4.0 expanded-query-search work it merges after: `candidate_strategy` (nearest/random/mixed), independent corpus scope (`t_start`/`t_end` + separate corpus filter), `--dtype fp32|bf16|fp16` compute-dtype plumbing
> > - Also carries the MLflow logging/param-tracking fixes planned as v2.3.1 (train_ids feed, latched empty-store warning, eval tracking-URI order, `logger.exception` swaps); v2.3.1 never shipped as a version number
> >
> > </details>

</details>

> <details>
> <summary><b>v2.3 — Retrieval-forecast model &amp; vector store (tagged 2026-09-06)</b></summary>
>
> > <details>
> > <summary><b>v2.3.0 — Retrieval-forecast model, vector store dataset, experiment config</b></summary>
> >
> > - Vector store dataset: precomputed embedding columns in an in-memory (FAISS/matmul) index
> > - `RetrievalForecast` model with differentiable straight-through top-k retrieval over a candidate pool
> > - `retrieval_forecast` experiment DI config; additive only
> >
> > </details>

</details>

> <details>
> <summary><b>v2.1 — Clean pipeline refactor</b></summary>
>
> > <details>
> > <summary><b>v2.1.0 — Ordered step specs, per-column intensity controls, nested CLI</b></summary>
> >
> > - Integer clean `level` replaced with per-operation step specs; steps are an ordered list on the job
> > - Preprocess CLI surface rebuilt; length-filter sign-error regression test added
> > - One manual `--runtime modal` dry run still pending
> >
> > </details>

</details>

> <details>
> <summary><b>v2.0 — Training &amp; evaluation on Modal</b></summary>
>
> > <details>
> > <summary><b>v2.0.0 — Training and evaluation on Modal</b></summary>
> >
> > - Venue-independent `TrainJob`/`EvalJob` pipelines; experiments build from `(runtime, env)` alone
> > - Client-side MLflow run creation, resumed in-place by `run_id` on both venues
> > - Fire-and-forget `.spawn()` train/eval on `ModalTrainingGPU`; checkpoints as MLflow artifacts enable cross-runtime resume
> > - Eval exports as `exports/<year>` MLflow artifacts; headless engine/trackers in containers
> >
> > </details>

</details>

</details>

<details>
<summary><b>v1.x — Experiment architecture and runtime abstractions</b></summary>

> <details>
> <summary><b>v1.3 — Modal describe &amp; engineer</b></summary>
>
> > <details>
> > <summary><b>v1.3.0 — Modal runtime for describe and engineer</b></summary>
> >
> > - Runtime-agnostic `DescribeJob`/`EngineerJob` pipelines; both dispatch with `--runtime modal` in the `text-forecast-data` app
> > - Engineer writes `metadata.json` recording the producing runtime
> >
> > </details>

</details>

> <details>
> <summary><b>v1.2 — Local / Modal runtime split</b></summary>
>
> > <details>
> > <summary><b>v1.2.0 — Local / Modal runtime split</b></summary>
> >
> > - `modal` runtime backend, volume-backed data sources, Modal GPU embedder
> > - `preprocess` runs with `--runtime modal`
> >
> > </details>

</details>

> <details>
> <summary><b>v1.1 — Runtime abstractions &amp; robust preprocessing</b></summary>
>
> > <details>
> > <summary><b>v1.1.0 — Migrate remaining apps + robust preprocessing + local/Modal prep</b></summary>
> >
> > - `preprocess`/`describe`/`engineer` apps on `load_env(...)` with `[env]` CLI overrides; Phase-0 `__getattr__` shim removed
> > - Robust preprocessing: lazy embedder loading, pluggable models, CPU/CUDA
> > - `Runtime` and `DataSource` abstractions introduced
> >
> > </details>

</details>

> <details>
> <summary><b>v1.0 — Dependency-injected experiment configs</b></summary>
>
> > <details>
> > <summary><b>v1.0.0 — Dependency-injected experiment configs</b></summary>
> >
> > - Self-contained experiment modules under `src/config/experiments/`; generic `Experiment[T_Batch]`, `Engine`, `Strategy` protocol
> > - Checkpoint processor abstraction; tracker rewrite; basedpyright strict, Python 3.13
> >
> > </details>

</details>

</details>

<details>
<summary><b>v0.x — Prototyping era: preprocessing, evaluation, and CLI foundations</b></summary>

> <details>
> <summary><b>v0.7 — CLI &amp; config consolidation</b></summary>
>
> > <details>
> > <summary><b>v0.7.0 — CLI &amp; Config Consolidation</b></summary>
> >
> > - Config/env CLI overrides; train loop refactored into `/apps`; train/data/env configs split; dedicated loss/optimisation config
> >
> > </details>

</details>

> <details>
> <summary><b>v0.6 — Metric export &amp; visualisation</b></summary>
>
> > <details>
> > <summary><b>v0.6.0 — Model Eval Metric export for visualisation</b></summary>
> >
> > - Batch-resilient dataset dataclass; outputs associated with input row ids; structured JSON metric exports
> >
> > </details>

</details>

> <details>
> <summary><b>v0.5 — Dataset caching, descriptives, and efficiency</b></summary>
>
> > <details>
> > <summary><b>v0.5.3 — Efficiency, Control &amp; Clean update</b></summary>
> >
> > - MLflow checkpoint loading with parent-run args; dataset registry/formatting/kwargs via CLI/config; LR scheduler args in config; lowercase clean method
> >
> > </details>
> >
> > <details>
> > <summary><b>v0.5.2 — Metric tracker efficiency &amp; Model train checkpoint loading</b></summary>
> >
> > - Leaner best-accuracy calculation; model checkpoint loading in the train loop
> >
> > </details>
> >
> > <details>
> > <summary><b>v0.5.1 — Descriptives app</b></summary>
> >
> > - CLI-driven descriptives per column; frequency buckets; proportional class weights
> >
> > </details>
> >
> > <details>
> > <summary><b>v0.5.0 — Dataset ipc refactor</b></summary>
> >
> > - IPC-cached datasets for fast random access of OOM-sized data; named dataset organisation
> >
> > </details>

</details>

> <details>
> <summary><b>v0.4 — Evaluation CLI &amp; metric tracker</b></summary>
>
> > <details>
> > <summary><b>v0.4.5 — Preprocess &amp; Train app upgrade</b></summary>
> >
> > - Column-specific cleaning levels; drop/clear split; license filtering; filter-based partitioning speedup; model logged as artifact
> >
> > </details>
> >
> > <details>
> > <summary><b>v0.4.4 — Enhanced Metric calculation</b></summary>
> >
> > - Best-threshold accuracy with precision/recall; PR &amp; ROC AUC charts as MLflow artifacts
> >
> > </details>
> >
> > <details>
> > <summary><b>v0.4.3 — Update data loading &amp; handling</b></summary>
> >
> > - Specialised, memory-efficient load funcs; example counts in train/eval logging
> >
> > </details>
> >
> > <details>
> > <summary><b>v0.4.2 — Standardise cross-app arg parsing</b></summary>
> >
> > - Standardised start/end date arg parsing across apps
> >
> > </details>
> >
> > <details>
> > <summary><b>v0.4.1 — Metric Tracker Generalisation</b></summary>
> >
> > - Generalised tracker init and metric calc responsive to store name prefix
> >
> > </details>
> >
> > <details>
> > <summary><b>v0.4.0 — Eval CLI</b></summary>
> >
> > - Evaluation CLI app: config-driven, temporal scrub, MLflow tracking, JSON export for historical performance graphs
> >
> > </details>

</details>

> <details>
> <summary><b>v0.3 — Dataloader flexibility</b></summary>
>
> > <details>
> > <summary><b>v0.3.0 — Dataloader Flexibility</b></summary>
> >
> > - Concatenation of multiple string/token columns when serving dataloader examples
> >
> > </details>

</details>

> <details>
> <summary><b>v0.2 — Data pre-processing CLI</b></summary>
>
> > <details>
> > <summary><b>v0.2.0 — Data pre-processing CLI</b></summary>
> >
> > - Pre-processing notebook standardised into functions, orchestrated in a Typer CLI with per-step option flags
> >
> > </details>

</details>

</details>
