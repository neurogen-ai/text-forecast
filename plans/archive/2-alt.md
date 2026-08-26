# Plan 2-alt — Rented-GPU containers instead of Modal

> **Scope:** evaluate replacing the Modal backend (plan 2.0) with a generic
> rented-GPU container backend (Vast.ai, RunPod, Lambda Labs, ...) using
> Cloudflare R2 for datasets and checkpoints. Compare the effort against
> finishing plan 2.0 as written, and spell out the steps if we go ahead.
>
> **Scope guard:** single-GPU only, same as plan 2.0. Multi-GPU stays in
> plans/2.2.md regardless of which backend ships first.

---

## 1. Verdict

Finishing plan 2.0 on Modal is less work than switching to rented containers
now. The two options share almost everything, and the shared part is the hard
part.

What the two backends have in common (all currently missing from the code):

1. `run_train_pipeline` / `run_eval_pipeline` extracted from
   `apps/train.py` / `apps/eval.py` into runtime-agnostic functions.
2. `TrainJob` / `EvalJob` dataclasses carrying experiment bytes, `Env`, and
   resume state (plan 2.0 §7–8).
3. Client-created MLflow runs so local and remote execution log into the same
   run (plan 2.0 P6).
4. Headless `Engine` (`progress=None`) so the loop runs without rich bars
   (plan 2.0 §11).
5. A network-reachable MLflow tracking server. Today the default is
   `http://127.0.0.1:5000` and checkpoints write through a local
   `artifact_loc`. Neither works from any remote container, Modal included.
   This is a prerequisite of both options and is already planned in
   `plans/MLflow-Cloud.md`.

Item 5 blocks Modal too, so it belongs to whichever option ships first.

What Modal does for us that a rented-container backend would have to replace:

| Concern | Modal today | Rented container |
|---|---|---|
| Image | `pip_install_from_pyproject` + `add_local_dir` (working, see `modal_runtime.py`) | Own Dockerfile, own registry push, pull auth on the instance |
| Storage | Named volumes, mount paths match `DataSource.resolve()` | R2 buckets via boto3/s3fs/rclone, staging scripts, cache invalidation |
| Dispatch | `.spawn()` returns a FunctionCall id; survives client disconnect | Provision instance via provider API, inject env, start job, poll, stream logs |
| Failure | Platform retries, cold starts handled | Spot preemption mid-training, stale instances, manual requeue logic |
| Teardown | Automatic scaledown | Must shut down instances ourselves or pay idle time |
| Secrets | Modal secrets (plan 2.1 F-29) | Env injection per instance, key rotation |

None of this is research risk. It is all writable. But it is weeks of
undifferentiated infrastructure work, and the `ModalEmbeddingGPU`,
volume-mount, and remote-dispatch precedents already exist and function. The
remaining Modal-specific work in plan 2.0 is roughly the dispatch wrapper,
the training GPU class, and the checkpoint-volume mapping (§10 of the plan),
because the image/volume/GPU-class patterns are proven by the preprocess app.

The one strong argument for rented containers is cost: Vast-class GPUs run a
fraction of Modal's per-hour price, and long fine-tunes are exactly where
that compounds. That is a reason to add a second backend later, not to
abandon the nearly-working one first.

---

## 2. Recommended sequencing

1. **Do plan 2.0 §4–9 and §11 first, unchanged.** Pipeline extraction, job
   dataclasses, client-created runs, headless Engine. None of it mentions
   Modal except in comments. This is the bulk of 2.0 and it is reusable by
   every future backend.
2. **Do the R2 storage work early, independent of backend choice.**
   `S3CheckpointProcessor` (stubbed in
   `src/training/checkpointing/s3.py`), R2-backed MLflow artifacts, dataset
   staging. `plans/MLflow-Cloud.md` already points artifact storage at R2, so
   this pays off even if Modal stays the only executor forever.
3. **Finish the thin Modal dispatch (plan 2.0 §10)** to get v2.0 shipped.
4. **Add `VastRuntime` (or RunPod) later as a third `Runtime`
   implementation** behind the same protocol, when training hours make the
   price difference matter. With steps 1–3 done, this new plan reduces to
   sections 4–6 below.

If we skip steps 1–3 and jump straight to rented containers, we still do all
of step 1, plus everything in sections 4–6, with no working cloud backend
until the last piece lands.

---

## 3. Provider choice

Requirements: API-driven instance creation, docker-image support, per-hour
billing, SSH access, tolerable preemption behaviour.

- **Vast.ai** — cheapest, large supply, but interruptible pricing means jobs
  die and must resume from checkpoints. P3 (checkpoints as the canonical
  store) makes resumption possible, but expect to design for it from day one.
- **RunPod** — slightly pricier, cleaner REST API, pod templates, dedicated
  (non-interruptible) options. Probably the best effort-to-reliability ratio.
- **Lambda Labs / TensorDock** — viable; pick on price and API shape when the
  time comes.

The plan below assumes Vast.ai wording but keeps provider details behind a
small adapter so swapping costs one module.

---

## 4. Storage layer (R2)

New config:

```toml
[env.storage]
endpoint = "https://<account>.r2.cloudflarestorage.com"
bucket_datasets = "citef-datasets"
bucket_artifacts = "citef-artifacts"
```

Steps:

1. Implement `S3CheckpointProcessor` against boto3 with the R2 endpoint.
   Same contract as `MlflowCheckpointProcessor`: save/load checkpoints,
   experiment-file I/O, fail-fast `experiment_file_exists`. Keep MLflow
   artifacts canonical (P3); R2 is the backing store behind the tracking
   server and/or the direct checkpoint store.
2. Dataset staging: `maybe_upload` / `maybe_download` implementations that
   sync a staged dataset directory to `s3://citef-datasets/<name>/`. A small
   manifest (parquet list + mtimes) avoids re-uploading unchanged partitions.
3. In-container access: mount via s3fs/rclone at startup (simplest, matches
   the "paths look like paths" property of `Env`) or read directly through
   boto3. Mounting keeps `Env.raw_loc`/`staged_loc` semantics identical to
   the Modal volume layout.
4. Credentials travel as env vars injected at instance creation, never in
   `config.toml` baked into images.

---

## 5. Container image

One `Dockerfile` mirroring the Modal image definition:

```dockerfile
FROM python:3.12-slim
WORKDIR /root
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY src/ /root/src/
ENV PYTHONPATH=/root/src
```

Steps:

1. Build in CI (or locally) and push to GHCR, private.
2. Tag images with the git sha so a run records the exact code it used;
   `latest` only for smoke tests.
3. Instance entrypoint: rclone mount the R2 buckets, wait for health,
   execute the job command, upload logs, exit nonzero on failure so the
   supervisor can react.
4. Keep torch/CUDA versions pinned to whatever the local environment uses;
   document the mismatch risk otherwise.

This replaces plan 2.0 §10.3 and is the first place the extra cost shows up.

---

## 6. Dispatch (`VastRuntime`)

```python
class VastRuntime:
    supported_source_backends = {"r2"}

    def get_source(self, root, name) -> DataSource: ...
    def get_embedder(self, key, **kwargs) -> TextEmbedder: ...   # in-job HF embedder; no RPC batching in v1
    def maybe_upload(self, source, local_path) -> None: ...      # -> R2
    def maybe_download(self, source) -> Path: ...                # <- R2
    def run_preprocess(self, job) -> DataSource: ...
    def run_train(self, job: TrainJob) -> RunResult: ...
    def run_eval(self, job: EvalJob) -> RunResult: ...
```

Config:

```toml
[runtime.vast]
api_key_env = "VAST_API_KEY"
image = "ghcr.io/fnoble/citation-forecast:<sha>"
gpu = "RTX_4090"
max_price_hr = 0.40
disk_gb = 100
timeout = 86400
```

Dispatch flow for `run_train(job)`:

1. Validate the tracking URI is network-reachable (same check as plan 2.0
   §10.5).
2. Derive the container `Env` (R2 mount paths replacing volume mounts), same
   shape as plan 2.0 §7.3.
3. Create the MLflow run client-side (P6, identical to plan 2.0 §8.3).
4. Submit the instance via the provider API: search offers within
   `max_price_hr`, create with the pinned image, inject env vars
   (tracking URI, R2 keys, run id), disk sized from config.
5. Start the job over SSH: `python -m apps.train_worker <job.json>`, where
   `train_worker` deserialises `TrainJob` and calls
   `run_train_pipeline(job, progress=None)`. The job spec travels as JSON
   (experiment bytes inline, exactly like plan 2.0 §7.2).
6. Poll instance status; tail logs into `logs/`; on completion parse the
   `RunResult` the worker writes to a known location (R2 or stdout sentinel).
7. Terminate the instance in a finally-block. Idle-billing leaks are the
   classic rented-GPU failure mode.
8. Return `RunResult(run_id=..., status="spawned"|"completed", ...)`.

Failure handling, minimum viable version:

- Preemption detected (instance gone): report status "interrupted"; resume is
  manual via the existing `--load-id/--load-epoch` flags. Automated requeue
  is plan-2.1 material, ported to this backend.
- Instance never starts or fails health checks: terminate, surface the
  provider error, do not silently retry into a bad offer.

Embedding during remote preprocess: simplest correct version embeds inside
the same container with the local HuggingFace embedder. The Modal split
(CPU dataframe container + batched GPU embedder class) was an optimisation;
skip it until profiling says otherwise.

---

## 7. Verification

Mirrors plan 2.0 §13:

```bash
# Local, unchanged
citef train -s smoke --no-gpu --subsample 512

# Rented-GPU training
citef --experiment graph_embed_class train -s smoke --subsample 512 --runtime vast

# Cross-runtime resume in both directions (P3)
citef train -s resume --load-id <remote-run-id> --load-epoch <n>
citef train -s resume --load-id <local-run-id> --load-epoch <n> --runtime vast

# Kill the instance mid-run; verify resume from the last checkpoint works
```

Expected: checkpoints land in R2/MLflow, metrics stream to the client-created
run, resume works across venues with no new code paths, and the instance is
always terminated afterwards.

---

## 8. Effort comparison

| Chunk | Finish plan 2.0 (Modal) | This plan (rented) |
|---|---|---|
| Pipeline extraction, jobs, client-created runs, headless Engine (§4–9, §11) | required | required, identical |
| Remote-reachable MLflow + artifact store | required | required, identical |
| Image | ~done (reuse preprocess pattern) | new Dockerfile + CI + registry |
| Storage | volumes exist; map checkpoint volume | new S3CheckpointProcessor + staging sync |
| Dispatch | thin wrapper over working patterns | full lifecycle: provision, exec, poll, teardown |
| Interruption/retry | platform-provided | ours, even if minimal |
| Secrets | Modal secrets (2.1) | env injection + rotation |

Rough sizing relative to each other: the shared chunk dominates both plans.
On top of it, Modal adds days; a rented backend adds one to two weeks of
infrastructure that produces no modelling capability.

## 9. Recommendation

Finish plan 2.0 as written, pulling section 4 (R2 storage) forward since
`MLflow-Cloud.md` needs it anyway. Revisit this plan when monthly GPU spend
makes the price gap real; at that point `VastRuntime` is a contained addition
to a stable protocol rather than a rewrite of the cloud story.
