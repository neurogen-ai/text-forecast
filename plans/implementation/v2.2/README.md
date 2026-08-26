# Plan 2.2 implementation — Modal runtime tidy-up

Implements `plans/2.2.md`. Refined against the code as of plan 2.1 T5
(commit `4a268ed`). One file per branch in this directory; each step fits one
context window and lands as one commit.

## Context

The 2.0 runtime branches left `runtime/modal_runtime.py` padded with noise
logging, trivial wrappers, volume labels spelled twice (module constant plus
`[runtime.modal]` config), and the `<experiment>-EVAL` naming derived
independently in three places. Input validation in the CLI apps still uses
bare `assert`, which vanishes under `python -O`. Plan 2.1 already rebuilt the
preprocess CLI and pipeline (`steps.py`, HANDLERS dispatch), so the old
assert sites it named in `apps/preprocess.py` are gone; the real remaining
assert sites are elsewhere (listed in branch D).

**Files touched:** `src/runtime/modal_runtime.py`, `src/runtime/local.py`,
`src/config/env.py` (read-only check), `src/training/pipeline/eval.py`,
`src/training/pipeline/_common.py`, `src/apps/train.py`,
`src/apps/eval.py`, `src/apps/engineer.py`,
`src/data/preprocess/pipeline.py`, `src/apps/preprocess.py`,
`tests/test_modal_config.py`, `tests/test_eval_naming.py` (new),
`tests/test_preprocess_job.py`, `docs/apps.md`.

**Depends on documents:** `plans/2.2.md` (the spec),
`plans/2.1-implementation.md` (context for what 2.1 already changed),
`docs/apps.md` (Modal dispatch and `[runtime.modal]` docs),
`docs/training.md` (eval flow).

## Deviations from plans/2.2.md found during inspection

The implementing agent does not need to reconcile these; they are already
reflected in the branch files.

1. The EVAL fallback is computed in **three** places, not two:
   `ModalRuntime.run_eval`, `LocalRuntime.run_eval`, and
   `run_eval_pipeline`. Branch C single-sources all three.
2. The assert sites named in 2.2 §1.5 moved with 2.1. The live assert sites
   are `apps/train.py` (3), `apps/eval.py` (6), `apps/engineer.py` (1),
   `training/pipeline/_common.py::build_run_context` (1),
   `training/pipeline/eval.py::_validate` (2), and an internal invariant in
   `data/preprocess/pipeline.py::_run_embed` (1).
3. The `POLARS_MAX_THREADS` question from 2.2 §2.6 is settled: setting the
   variable after polars import has no effect (verified against the project
   venv pin; thread pool is sized at import). Branch E deletes the dead
   line, the `max_threads` job field, and the CLI flag.
4. `modal_runtime_config()` raises on unknown keys, so using it at import
   time is safe: the container bakes the same `config/` directory as the
   client (`add_local_dir("config", ...)`), so any file that passes
   validation client-side passes in the container too.
5. The `VOLUME_LABEL` mismatch check compares two *config keys*
   (`[source.modal].volume` vs `[runtime.modal].staged_volume`) once the
   constants are gone. Branch B keeps the check with an updated message;
   deleting it entirely would let the two keys drift silently and mount the
   wrong volume.

## Goals

- Under 12 logger calls in `runtime/modal_runtime.py`, each carrying state a
  reader cannot get from return values.
- No trivial wrappers or duplicated constants in `runtime/modal_runtime.py`.
- Exactly one owner of the `<experiment>-EVAL` / `eval-<run_id>` naming.
- Zero `assert` statements used as input validation in `src/`.
- Dead `POLARS_MAX_THREADS` plumbing removed.
- No behaviour change beyond the deletions above: same experiment names,
  same volumes mounted, same error messages where tests match on them.

## Success criteria

- `pytest tests/ -q` green after every branch.
- `grep -c "logger\." src/runtime/modal_runtime.py` returns ≤ 12.
- `grep -rn "assert " src/apps src/training src/data/preprocess/pipeline.py`
  returns no input-validation asserts.
- `grep -n "VOLUME_LABEL = \"" src/runtime/modal_runtime.py` matches only the
  embedders label (infra path constant, see branch B).
- `-EVAL` naming defined once, in `src/training/pipeline/eval.py`.
- `docs/apps.md` claims verified against code (branch F checklist).

## Tests

Existing suite is the regression net: `pytest tests/ -q`. New tests per
branch are listed in the branch files. Run pyright on changed modules:
`pyright <files>` (config in `pyrightconfig.json`).

## Branch map

```
main ──┬── A  chore/modal-runtime-tidy        logging cut + wrapper inline   (modal_runtime.py)
       │    └──┬── B  refactor/volume-single-source   [after A, same file]
       │       └── C  refactor/eval-naming-helper     [after A, same file]
       │            (B and C touch disjoint regions of modal_runtime.py;
       │             land in either order, rebase whichever merges second)
       ├── D  refactor/asserts-to-raises       [no deps]
       └── E  chore/drop-dead-polars-threads   [no deps]
            (D and E both touch data/preprocess/pipeline.py in different
             hunks; whoever lands second rebases)

       F  docs/v2.2-closeout               [after A–E all merged]
```

- **Bottleneck:** A blocks B and C (same file). Nothing else blocks anything.
- **Parallel from main:** A, D, E can all start immediately in parallel.
- B, C, D, E are independent features; each forms one PR.

## Logical commits

| Branch | Commits |
|--------|---------|
| A | `chore(runtime): cut modal logging to dispatch boundaries`; `refactor(runtime): inline trivial wrappers in modal_runtime` |
| B | `refactor(runtime): single-source modal volume labels from [runtime.modal]` |
| C | `test(eval): pin eval experiment naming`; `refactor(eval): single-source -EVAL naming via eval_experiment_name` |
| D | `refactor(pipeline): explicit raises in train/eval pipelines`; `fix(cli): replace asserts with BadParameter in apps` |
| E | `chore(preprocess): remove dead POLARS_MAX_THREADS plumbing` |
| F | `docs: verify apps/training docs against 2.2 reality; mark plan implemented` |

## Abstractions introduced, and what each absorbs

- **`eval_experiment_name(job) -> str`** in `training/pipeline/eval.py`.
  Absorbs any future naming change (a rename scheme, a new override field,
  a config-driven prefix): the change lands in one function, not three runtimes.
- **Import-time `_MODAL_CFG = modal_runtime_config(_ENV)` boundary** in
  `modal_runtime.py`. Absorbs new `[runtime.modal]` keys (new GPU roles,
  snapshot toggles, timeouts): add a field to `ModalRuntimeConfig` in
  `config/env.py`, read it here once. Callers never learn where config comes from.
- **No abstraction for assert replacement.** Each guard stays inline at its
  site. A shared "validate" helper would hide which app raised and add a
  module to learn; the guards differ per command and will not grow together.
