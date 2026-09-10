# Roadmap

Forward-looking plans for forecite, in planned execution order. Completed
versions live in [CHANGELOG.md](CHANGELOG.md); the authoritative
shipped-vs-pending status table is
[`plans/releases/README.md`](plans/releases/README.md), and each entry below
links its full plan doc there.

Near term: v2.1.0 is implemented and awaiting one manual `--runtime modal`
dry run to close out ([plan 2.1](plans/releases/archive/2.1.md)).

<details>
<summary><b>v2.x — Cloud runtime maturity &amp; production readiness</b></summary>

> <details>
> <summary><b>v2.5 — Modal runtime tidy-up &amp; preprocess resume</b></summary>
>
> - Cut Modal dispatch logging to one INFO line per boundary; inline trivial wrappers
> - Single-source volume labels from `[runtime.modal]` (delete the hardcoded constant)
> - One owner for the `<experiment>-EVAL` naming; `assert`-based validation moved to explicit raises
> - Carries the preprocess id-skip resume remainder: sorted input glob, skip-if-empty guard, `--resume` flag (proposal v2.11)
> - Eval dispatch clears `torch.compile` under Modal GPU snapshots, mirroring the train path (eval currently compiles unconditionally on CUDA)
> - Tokeniser path resolved against the project root instead of CWD (containers re-download per run today); 1h preprocess timeout constraint recorded
> - Gumbel random selection made user-configurable on the retrieval-forecast B model: a random-proportion knob (0–1) governing the Gumbel-random share of the candidate pool, and an exposed tau constant smoothing the draw (replacing the hardcoded `MIXED_RANDOM_RATIO`/backend-only `GUMBEL_TAU`; subsumes the rejected v2.10 random-share proposal)
> - Status: drafted; implementation plan written (branches A–F), not started; T8 in progress on branch `v2.5-retrieval-forecastB`
>
> </details>
>
> <details>
> <summary><b>v2.6 — Production deployment hardening</b></summary>
>
> - CI smoke tests (typecheck + tests per PR) with no Modal credentials
> - Central Modal image builder; pin deployed images via `[runtime.modal].image_tag`; deploy script + tag-based CI deploys
> - Secrets management: config carries secret *names* only, values never enter `config.toml`
> - `status` CLI command to follow spawned jobs by id, with dispatch retry on transient failure
> - Versioned data volumes; incremental preprocess runs over new raw partitions only
> - Opt-in scheduled retrain and web endpoints (severable cut point if the release slips)
> - Docs sweep and plan closeout
> - Status: not started
>
> </details>
>
> <details>
> <summary><b>v2.7 — Distributed GPU training &amp; eval</b></summary>
>
> - Multi-GPU with one knob: `--gpus 2` locally, `train_gpu = "A10G:4"` on Modal
> - `world_size`/`rank` flow through `RunContext`; experiment files unchanged
> - DDP wrap with rank-0-only checkpoints, MLflow logging, and progress; epoch-boundary all-gather of metric stores
> - Single-process (`--gpus 1`) behaviour stays bit-for-bit identical
> - Status: not started; multi-node clusters are an explicit non-goal (v3.0 candidate)
>
> </details>
>
> <details>
> <summary><b>v2.8 — Remote progress tracking</b></summary>
>
> - `Reporter` interface replaces the `progress=None` guards: spawned Modal runs batch progress updates and flush them to MLflow, so long detached train/eval runs are followable while executing
> - Status: plan file reconstructed after the original was lost (2026-06-05) — review before executing
>
> </details>
>
> <details>
> <summary><b>v2.9 — CLI launch speed</b></summary>
>
> - Defer heavy third-party imports (torch, transformers, mlflow, seaborn, sklearn) out of module scope; `--help` and every dispatch boundary become responsive
> - Modal client paths never import torch
> - Status: drafted; implementation plan written (branches A–E), not started
>
> </details>

</details>