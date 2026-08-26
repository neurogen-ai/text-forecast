# Branch B — `refactor/volume-single-source`

Deletes the hardcoded volume-label constants in
`src/runtime/modal_runtime.py` and reads every label from
`[runtime.modal]` via `modal_runtime_config`. Spec: `plans/2.2.md` §2.3.
Depends on branch A (same file). Parallel with C, D, E once A lands.

## Step B1 — Import-time config boundary replaces the constants

**Files:** `src/runtime/modal_runtime.py`, `tests/test_modal_config.py`.

**Contract after this step:**

```python
# module level, after _ENV = load_env()
_MODAL_CFG = modal_runtime_config(_ENV)
_STAGED_VOLUME_LABEL = _MODAL_CFG.staged_volume
_EMBEDDERS_VOLUME_LABEL = "embedders"   # infra label: no config key exists
_CHECKPOINT_VOLUME_LABEL = (
    _MODAL_CFG.checkpoint_volume or "cf-checkpoints"
)
```

- `VOLUME_LABEL` and the `STAGED_VOLUME_LABEL = _MODAL_CONFIG.get(...)` line
  are deleted. `_STAGED_VOLUME_LABEL` is the only spelling of the staged
  label; it feeds `volume`, the three remote-function decorator
  `volumes={f"/modal/{_STAGED_VOLUME_LABEL}": volume}` entries, and
  `train_staged_volume`.
- The `or "cf-checkpoints"` fallback stays, with a one-line comment pointing
  at the existing rationale above it (the container re-imports this module
  with baked config and must not crash on a missing key; client-side train/
  eval dispatch still fails fast through
  `modal_runtime_config(require_checkpoint_volume=True)`).
- `EMBEDDERS_VOLUME_LABEL` stays a literal. It has no config counterpart;
  inventing a config key for it is out of scope.
- `ModalRuntime.__init__` keeps its check but compares the two config keys,
  not a constant:

  ```python
  cfg = modal_runtime_config(env)
  configured_volume = env.source.get("modal", {}).get("volume")
  if configured_volume != cfg.staged_volume:
      raise ValueError(
          "[source.modal].volume must equal [runtime.modal].staged_volume "
          f"({cfg.staged_volume!r}); got {configured_volume!r}."
      )
  ```

  Behaviour note: when `[source.modal]` is absent this raises exactly as the
  old code did (`None != label`). When both keys agree, nothing changes.
- Update the module docstring comment block that says "Volume labels ... are
  module constants because Modal function decorators are evaluated at import
  time" to say labels come from `[runtime.modal]` at import time.

**Test changes in `tests/test_modal_config.py`:** add

```python
def test_source_and_runtime_volume_mismatch_raises() -> None:
    env = Env(
        tracking_uri="http://127.0.0.1:5000",
        artifact_loc=Path("/tmp/artifacts"),
        runtime={"default": "local", "modal": {"staged_volume": "stage-a"}},
        source={"default": "local", "modal": {"volume": "stage-b"}},
    )
    from runtime.modal_runtime import ModalRuntime
    with pytest.raises(ValueError, match="must equal"):
        ModalRuntime(env, project="text-forecast")
```

(match on `"must equal"` so the test survives message-wording tweaks). If
importing `runtime.modal_runtime` is too heavy for this suite, put the test
in a new `tests/test_modal_runtime_init.py` instead; do not skip it.

**Out of scope for this step:** removing `[source.modal].volume` as a config
key (the source-backend abstraction reads per-backend dicts generically;
changing that belongs to a source-config plan), renaming volumes in
`config/config.toml`, touching `container_env` in `runtime/modal_env.py`
(it already derives mounts from `cfg.staged_volume`), deleting
`raw_volume` from `ModalRuntimeConfig` even if nothing reads it yet.

**Verify:** `.venv/bin/python -m pytest tests/test_modal_config.py -q` green;
`grep -n "VOLUME_LABEL\|VOLUME = " src/runtime/modal_runtime.py` shows only
`_STAGED_VOLUME_LABEL`, `_EMBEDDERS_VOLUME_LABEL`,
`_CHECKPOINT_VOLUME_LABEL`;
`grep -rn '"openalex-staged"' src/runtime/` returns nothing.
Commit: `refactor(runtime): single-source modal volume labels from [runtime.modal]`.

## Step B2 — Config example and docs consistency check

**Files:** `config/config.example.toml` (only if needed).

**Changes:** confirm `[runtime.modal].staged_volume` appears in
`config.example.toml`; add it next to the existing keys if missing (copy the
value from `config/config.toml`, `"openalex-staged"`). Confirm
`docs/apps.md` §"Modal config" still lists `staged_volume` and does not
mention module constants; fix any sentence that says the label is a constant
in `modal_runtime.py`.

**Out of scope:** rewriting docs/apps.md (branch F owns the full sweep).

**Verify:** `.venv/bin/grep -c staged_volume config/config.example.toml docs/apps.md`
returns ≥ 1 each; no code changes in this commit except the example toml.
Commit: folded into the B1 commit if the example already had the key;
otherwise `docs(config): document [runtime.modal].staged_volume in example`.
