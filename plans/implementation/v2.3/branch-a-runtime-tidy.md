# Branch A — `chore/modal-runtime-tidy`

Cuts `src/runtime/modal_runtime.py` from 33 logger calls to 11 and deletes
the trivial wrappers. No logic change. Spec: `plans/2.3.md` §2.1, §2.3.
Depends on nothing; blocks branches B and C (same file).

## Step A1 — Logging cut to dispatch boundaries

**Files:** `src/runtime/modal_runtime.py` only.

**Target:** exactly these 11 logger calls, no others.

Keep:

1. Module-level DEBUG config summary (line ~66), unchanged. It already
   prints gpu, python_version, embedder_batch_size, checkpoint_volume,
   train_gpu, timeout in one call.
2. `ModalEmbeddingGPU.setup`: one INFO **after** the embedder loads:
   `"ModalEmbeddingGPU.setup: embedder %r loaded on %r"` with
   `self.model_name` and `_GPU_TYPE`. Delete the "loading" line, the
   "creating HuggingFaceEmbedder" line, the `embedder.load()` debug line,
   and the "loaded successfully" line.
3. `ModalTrainingGPU.setup`: keep the existing one-line INFO.
4. `ModalTrainingGPU.train`: change the entry line to include the run id:
   `logger.info("ModalTrainingGPU.train: run=%r", job.run_id)`.
5. `ModalTrainingGPU.eval`: same pattern with `job.run_id`.
6–8. `run_preprocess_remote`, `run_describe_remote`, `run_engineer_remote`:
   keep only the existing entry INFO lines (origin/destination, source,
   origin/output). Delete every `... job completed` line and the
   `run_preprocess_remote` steps debug line. Completion is visible from the
   function return and per-partition write logs already emitted by the
   pipeline.
9. `ModalRuntime.run_train`: delete the "spawning" line; keep the spawned
   line (`run_id`, `_TRAIN_GPU`, `call.object_id`) as the single dispatch
   signal. Keep the torch.compile warning unchanged; it is a real behaviour
   notice.
10. `ModalRuntime.run_eval`: same shape as run_train: delete "spawning",
    keep the spawned line with `eval_run_id` and `call.object_id`.
11. Nothing else. Specifically delete:
    - both `ModalEmbeddingGPU.encode` debug lines,
    - `_InContainerRuntime.get_embedder` debug line,
    - `_InContainerRuntime.run_preprocess` INFO line (its only caller,
      `run_preprocess_remote`, logs the entry),
    - the three client-side `ModalRuntime.run_*` "dispatching"/"received
      result" pairs around `fn.remote(job)` (six calls gone; Modal's own
      output via `modal.enable_output()` shows activity, and callers such as
      the apps log the outcome),
    - all three `_build_modal_embedder` debug lines (gone entirely in A2).

**Out of scope for this step:** deleting or renaming any function; touching
the volume constants; changing log levels in any other module; changing what
`setup_logger` does.

**Verify:** `.venv/bin/python -m pytest tests/test_runtime_dispatch.py tests/test_modal_config.py -q`
green; `grep -c "logger\." src/runtime/modal_runtime.py` prints a number ≤ 12;
`.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import runtime.modal_runtime"`
imports without error. Commit: `chore(runtime): cut modal logging to dispatch boundaries`.

## Step A2 — Inline the wrappers

**Files:** `src/runtime/modal_runtime.py` only.

**Contracts after this step:**

- Deleted functions: `_in_container_runtime()`, `_key_to_model_name(key)` ,
  `_embedder_config(key)`, `_resolve_output_dim(model_name)`.
- The error message currently built by `_embedder_config` must survive verbatim,
  now raised from `_build_modal_embedder`:
  ```
  f"Modal runtime does not support embedder {key!r}. "
  f"Supported keys: {list(EMBEDDERS)}"
  ```
  raised as `ValueError` when `key` is not in `EMBEDDERS` **and** `kwargs`
  supplies no `model_name`. Note the current code raises whenever the key is
  missing from EMBEDDERS even when `model_name` is passed; preserve that
  behaviour exactly (raise before reading `kwargs["model_name"]`) so unknown-key
  typos still fail loudly.
- The output-dim error must survive verbatim:
  ```
  f"Unknown output dimension for {model_name!r}. "
  "Add it to EMBEDDERS in src/runtime/modal_runtime.py."
  ```

**Changes:**

- Replace the single `_in_container_runtime()` call inside
  `run_preprocess_remote` with `_InContainerRuntime()`; delete the function
  and move its docstring's first sentence onto `_InContainerRuntime`.
- Rewrite `_build_modal_embedder(key: str, **kwargs: Any) -> TextEmbedder`
  to do, in order: registry lookup with the preserved ValueError; model name
  resolution (`kwargs.get("model_name") or config.model_name`); output dim
  scan over `EMBEDDERS.values()` matching `config.model_name == model_name`
  with the preserved ValueError; batch size
  (`int(kwargs.get("batch_size") or _EMBEDDER_BATCH_SIZE)`); construct and
  return `ModalEmbedder(model_name=..., output_dim=..., batch_size=...,
  gpu_cls=ModalEmbeddingGPU)`. No logging (A1 removed it).
- Update the docstring comment on `_resolve_output_dim`'s old message if it
  references the deleted helper.

**Out of scope for this step:** moving `EMBEDDERS` itself, changing
`ModalEmbedder`, touching `data/preprocess/embed_huggingface.py`, adding
tests for embedder construction (existing suite covers dispatch paths).

**Verify:** `.venv/bin/python -m pytest tests/ -q` green;
`.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import runtime.modal_runtime"`;
`grep -n "_in_container_runtime\|_key_to_model_name\|_resolve_output_dim\|_embedder_config" src/runtime/modal_runtime.py`
returns nothing. Commit: `refactor(runtime): inline trivial wrappers in modal_runtime`.
