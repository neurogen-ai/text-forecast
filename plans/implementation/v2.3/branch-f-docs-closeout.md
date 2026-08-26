# Branch F — `docs/v2.3-closeout`

Verifies documentation claims against post-2.3 reality and marks the plan
implemented. Runs last, after A–E merge.

## Step F1 — Docs sweep

**Files:** `docs/apps.md`, `docs/training.md`, `plans/README.md`.

**Checklist (fix each doc where a claim is now false; otherwise leave it):**

- `docs/apps.md` §"Modal config": keys list matches `ModalRuntimeConfig`
  fields; no sentence claims volume labels are module constants in
  `modal_runtime.py`; `[source.modal].volume` documented as "must equal
  `[runtime.modal].staged_volume`".
- `docs/apps.md` eval section: `<experiment>-EVAL` naming described without
  naming a specific owner function; if it names one, it must say
  `eval_experiment_name` in `training/pipeline/eval.py`.
- `docs/training.md`: nothing asserts about logger call counts or the
  `max_threads` job field.
- Neither doc documents `_in_container_runtime`, `_embedder_config`,
  `_key_to_model_name`, or `POLARS_MAX_THREADS`; delete any stale mention.
- `plans/README.md`: mark 2.3 as implemented with the same phrasing style as
  the 2.1 row; note anything from the manual checklist that did not run.
- Move nothing to `plans/archive/` yet: archive moves happen when the release
  ships, matching how 2.1 was handled.

**Out of scope for this step:** rewriting docs sections, touching README.md
quickstart unless it mentions deleted flags (`--max-threads`) or constants,
updating plans/2.3.md itself.

**Verify:** read each edited section against the code, not memory;
`.venv/bin/python -m pytest tests/ -q` green (docs-only commit, suite should
be untouched); `grep -rn "max_threads\|VOLUME_LABEL\|_in_container_runtime\|_key_to_model_name" docs/ plans/README.md`
returns nothing. Commit: `docs: verify apps/training docs against 2.3 reality; mark plan implemented`.

## Step F2 — Manual modal smoke (checklist item)

Same caveat as plan 2.1's pending item: this spins up paid Modal containers.
Run once when cost is acceptable:

```
PYTHONPATH=src .venv/bin/python -m main preprocess <small-origin> \
  --runtime modal --dry-run --op "clean:text" --trim-min-chars 20
```

Confirm: dispatch succeeds, container logs show exactly one INFO line per
remote entry point, and the job pickles cleanly (no `max_threads` attribute
error on the receiving side). Record the outcome in `plans/README.md`'s 2.3
row. If skipped, leave an explicit note rather than marking it done.
