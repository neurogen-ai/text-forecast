# Branch D — `refactor/asserts-to-raises`

Converts every input-validation `assert` in `src/` to explicit raises with
the identical message text. Spec: `plans/2.2.md` §2.5, updated for the sites
that exist after plan 2.1 (`apps/preprocess.py` is already assert-free).
Depends on nothing; parallel with A, B, C, E.

## Step D1 — Pipeline-layer raises

**Files:** `src/training/pipeline/eval.py`,
`src/training/pipeline/_common.py`, `src/data/preprocess/pipeline.py`.

**Changes (mechanical; keep each message byte-identical):**

- `eval.py::_validate`: both asserts become
  `raise ValueError(...)` with the same messages:
  `"Provide an interval"` and the f-string
  `f"Interval unit must be one of: {list(_T_DELTA_MAP.keys())}"`.
- `_common.py::build_run_context`: the GPU guard becomes
  `raise ValueError("No GPU available on this venue, use --no-gpu")`
  under the same condition (`device.type != "cuda" and gpu`).
- `data/preprocess/pipeline.py::_run_embed` (~line 118): the internal
  invariant becomes
  `raise ValueError("EmbedStep requires an embedder on StepContext")`.
  This is a programmer-error guard, not user input; ValueError keeps it
  visible under `python -O`.

**Out of scope for this step:** the CLI apps (D2), adding new validation
rules, changing any condition, wrapping errors, adding logging to failure
paths, touching `apps/train.py`'s or `apps/eval.py`'s bodies beyond what D2
lists.

**Verify:** `.venv/bin/python -m pytest tests/test_pipeline_jobs.py tests/test_batch_conformance.py tests/test_headless_engine.py -q`
green; then full `.venv/bin/python -m pytest tests/ -q`. Add no new test file
in this commit; D3 adds the assertion-type test.
Commit: `refactor(pipeline): explicit raises in train/eval pipelines`.

## Step D2 — CLI apps: BadParameter

**Files:** `src/apps/train.py`, `src/apps/eval.py`, `src/apps/engineer.py`.

**Changes:** replace each assert at the top of the command body with
`raise typer.BadParameter(<same message>)`. Conditions stay identical,
including the odd-looking ones in `apps/eval.py`:

- `assert start_date or not interval` → raise unless `start_date or not interval`
- `assert interval or not end_date` → likewise

All nine asserts across the three files go; `typer` is already imported in
each.

**Out of scope for this step:** reordering validation relative to
`load_env()`, merging guards, changing help text, touching flag defaults.

**Verify:** `.venv/bin/python -m pytest tests/ -q` green; manual smoke of one
error path:
`.venv/bin/python -m apps.eval 2>&1 | head -5` still exits with usage output
(no traceback from an AssertionError anywhere).
Commit: `fix(cli): replace asserts with BadParameter in apps`.

## Step D3 — Prove raises survive `python -O`

**Files:** `tests/test_validation_raises.py` (new).

**Changes:** three focused tests that call the guarded functions directly and
match exception **type**, so an `AssertionError` regression fails:

```python
import pytest

def test_eval_validate_requires_interval() -> None:
    job = _minimal_job(interval=None)   # build EvalJob as in test_pipeline_jobs.py
    with pytest.raises(ValueError, match="Provide an interval"):
        eval_module._validate(job)

def test_build_run_context_requires_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="No GPU available"):
        build_run_context(gpu=True)

def test_run_embed_requires_embedder() -> None:
    ctx = StepContext(runtime=..., embedder=None, partition_index=0, logger=...)
    with pytest.raises(ValueError, match="embedder on StepContext"):
        _run_embed(empty_lf, EmbedStep(...), ctx)
```

For the third test, `_run_embed` raises before any polars work, so an empty
`pl.LazyFrame()` schema-less frame and a minimal `EmbedStep` suffice; check
`tests/test_steps.py` for how steps are constructed and copy it. If mocking
`torch.cuda.is_available` proves awkward, run the second test under
`subprocess` with `PYTHONHASHSEED` untouched and simply assert non-zero exit;
do not skip the type assertion for tests 1 and 3.

Also run once with optimisation enabled as a manual check:
`.venv/bin/python -O -m pytest tests/test_pipeline_jobs.py -q` green.

**Out of scope for this step:** property tests, hypothesis, covering every CLI
flag combination.

**Verify:** `.venv/bin/python -m pytest tests/test_validation_raises.py -q`
green; full suite green.
Commit: `test(validation): raises not asserts survive python -O`.
