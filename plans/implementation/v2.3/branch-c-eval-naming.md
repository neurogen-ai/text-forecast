# Branch C — `refactor/eval-naming-helper`

Single-sources the eval MLflow experiment naming into one helper and calls
it from all three current sites. Spec: `plans/2.3.md` §2.4, extended to cover
`LocalRuntime` (found during inspection). Depends on branch A (edits
`runtime/modal_runtime.py`). Parallel with B, D, E once A lands.

## Step C1 — Pin the naming with a test

**Files:** `tests/test_eval_naming.py` (new).

**Contract under test:**

```python
def eval_experiment_name(job: EvalJob) -> str:
    """Final MLflow experiment for an eval run.

    Explicit ``job.experiment_name`` wins; otherwise
    ``<training-experiment>-EVAL`` when ``job.module_name`` resolves;
    otherwise ``eval-<run_id>``.
    """
```

Write the tests against `eval_experiment_name` **before** it exists; they fail
on import, which is the point. Cases:

1. override: `EvalJob(run_id="r", epoch=0, start_date=datetime(2020,1,1),
   module_name="", experiment_name="custom", env=...)` → `"custom"`.
2. derived: `module_name` set to a known experiment module (use the same real
   module fixture approach as `tests/test_runtime_dispatch.py`, which imports
   a real config experiments module) whose `read_experiment_name` returns
   `"transformer_class"` → `"transformer_class-EVAL"`.
3. fallback: `module_name=""` → `"eval-r"`.
4. unknown module: `module_name="does-not-exist"` → expect the
   `ValueError("Unknown experiment ...")` raised by `read_experiment_name`
   propagates (naming fails fast client-side, matching today's behaviour).

Env construction can reuse whatever `tests/test_pipeline_jobs.py` does for
`EvalJob`; check that file first and copy its pattern.

**Verify:** `.venv/bin/python -m pytest tests/test_eval_naming.py -q` fails on
ImportError only. Commit: `test(eval): pin eval experiment naming`.

## Step C2 — Add the helper, swap all three call sites

**Files:** `src/training/pipeline/eval.py`,
`src/runtime/local.py`, `src/runtime/modal_runtime.py`.

**Changes:**

- In `src/training/pipeline/eval.py`, add `eval_experiment_name(job: EvalJob)
  -> str` next to `_validate`, implemented exactly per the C1 docstring:

  ```python
  def eval_experiment_name(job: EvalJob) -> str:
      if job.experiment_name:
          return job.experiment_name
      training = (
          read_experiment_name(job.module_name) if job.module_name else None
      )
      return f"{training}-EVAL" if training else f"eval-{job.run_id}"
  ```

  Note `if training is not None` is required over truthiness only if an
  experiment name could be empty; `read_experiment_name` never returns empty,
  so plain truthiness is fine. Keep the existing comment about when each arm
  applies.
- In `run_eval_pipeline`: replace the inline derivation of `mlflow_experiment`
  with `mlflow_experiment = eval_experiment_name(job)`. Keep the separate
  `training_experiment = read_experiment_name(job.module_name) if job.module_name else None`
  line; it locates checkpoint artifacts for `MlflowCheckpointProcessor` and is
  a different concern from naming.
- In `LocalRuntime.run_eval` (`src/runtime/local.py`): replace the inline
  block computing `eval_experiment` with
  `eval_experiment = eval_experiment_name(job)` plus the import. Delete the
  now-unused local `training_experiment` there if nothing else in the method
  uses it.
- In `ModalRuntime.run_eval` (`src/runtime/modal_runtime.py`): same
  replacement. The method keeps `training_experiment` only if still needed
  after the swap; delete it if unused.

Byte-for-byte guarantee: for every input shape, the new helper returns the
same string as the old inline code at all three sites (same arms, same order).
The C1 test locks arms 1–4.

**Out of scope for this step:** changing `EvalJob` fields (`experiment_name`
override and `module_name` stay), renaming the `-EVAL` suffix, touching
`create_run` or `run_lifecycle`, adding the name to `RunResult`.

**Verify:** `.venv/bin/python -m pytest tests/test_eval_naming.py tests/test_runtime_dispatch.py tests/test_pipeline_jobs.py -q`
green; full suite `.venv/bin/python -m pytest tests/ -q` green;
`grep -rn '\-EVAL' src/ --include="*.py"` returns only
`src/training/pipeline/eval.py`. Commit: `refactor(eval): single-source -EVAL naming via eval_experiment_name`.
