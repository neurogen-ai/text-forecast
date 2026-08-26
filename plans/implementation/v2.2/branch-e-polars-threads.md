# Branch E — `chore/drop-dead-polars-threads`

Removes the dead `POLARS_MAX_THREADS` plumbing. The question 2.2 §2.6 left
open is settled: polars sizes its thread pool at import, so setting the env
var inside `run_preprocess_pipeline` does nothing (verified against the
project venv pin: `pl.thread_pool_size()` stays unchanged after the set).
Depends on nothing; parallel with A, B, C, D.

## Step E1 — Delete the plumbing

**Files:** `src/data/preprocess/pipeline.py`, `src/apps/preprocess.py`,
`tests/test_preprocess_job.py`.

**Contract changes:**

- `PreprocessJob` loses the field `max_threads: int = 8`
  (`src/data/preprocess/pipeline.py`). This is a job-serialisation format
  change: jobs pickled before this commit fail to unpickle afterwards.
  Acceptable by plan 2.2's own terms ("no behaviour that matters"); note it
  in the commit message. There are no persisted queues; jobs live for one
  dispatch.
- `run_preprocess_pipeline` loses its first line
  `os.environ["POLARS_MAX_THREADS"] = f"{job.max_threads}"`. If `os` has no
  other user in that module after deletion, drop the import too (check:
  nothing else in the file uses `os.environ`/`os.makedirs`... `os.makedirs`
  exists, so keep the import).
- `apps/preprocess.py` loses the `max_threads` typer option and the
  `max_threads=max_threads` argument in the `PreprocessJob(...)` call. The
  flag disappears from `ctx.params`, so `metadata.json` written by the
  pipeline stops recording `max_threads`; intentional, mention in the commit.
- `tests/test_preprocess_job.py`: remove `max_threads=4` from the fixture
  job (~line 100) and any assertion reading it.
- Keep `TOKENIZERS_PARALLELISM` in `apps/preprocess.py` as-is but add the
  one-line comment: `# Set before transformers/tokenisers first use; must
  precede model load, so it lives at import.`

**Out of scope for this step:** making thread count configurable for real
(e.g., a launcher-level env or a worker entry point); touching polars scan or
sink calls; any change to partitioning logic that currently divides rows by
`n_partitions`.

**Verify:** `.venv/bin/python -m pytest tests/test_preprocess_job.py tests/test_preprocess_e2e.py tests/test_runtime_dispatch.py -q`
green; full suite `.venv/bin/python -m pytest tests/ -q`;
`.venv/bin/text-forecast preprocess --help 2>/dev/null | grep -c max-threads || true`
prints 0 (or run the app entry point used by the project; the point is the
flag is gone from help output).
Commit: `chore(preprocess): remove dead POLARS_MAX_THREADS plumbing`.
