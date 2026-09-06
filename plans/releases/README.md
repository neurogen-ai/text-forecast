# Plans

Forward-looking release plans live here, one file per version. Completed plans
move to `archive/` untouched. Ops guides belong in `docs/`.

## Numbering convention

Plan numbers match release versions. Minor releases may contain multiple
independent refactors only if they ship together; otherwise each gets its own
number. Patch releases (x.y.z) are reserved for fixes with no user-visible
contract change.

## Active plans

| Plan | Title | Status |
|------|-------|--------|
| [2.1](2.1.md) | Clean pipeline refactor: ordered step specs, per-column intensity controls, nested CLI | implemented (T0–T5); pending one manual `--runtime modal` dry run |
| [2.2](2.2.md) | CLI launch speed: lazy heavy imports, torch-free modal client paths | drafted; [implementation plan](implementation/v2.2.md) written (branches A–E), not started |
| [2.3](2.3.md) | Retrieval-forecast model, vector store dataset, experiment config | implemented (spec written retroactively) |
| [2.3.1](2.3.1.md) | MLflow logging fixes: train_ids feed, latched empty-store warning, eval tracking-URI order, tracking logger.exception swaps | ratified (pending implementation) |
| [2.4](proposals/) | Expanded query search strategy: candidate pool selection (nearest/random/mixed), independent corpus scope, compute dtype plumbing | proposed; [proposal files](../proposals/) awaiting oracle-vetted ratify. Renumbered the drafted 2.4–2.7 chain to 2.5–2.8 to free the number |
| [2.5.0](2.5.0.md) | Modal runtime tidy-up: dedupe config/log noise, inline wrappers, single-source EVAL naming | drafted; [implementation plan](implementation/v2.5.0.md) written (branches A–F), not started |
| [2.6](2.6.md) | Production deployment hardening | not started |
| [2.7](2.7.md) | Distributed GPU training/eval | not started |
| [2.8](2.8.md) | Remote progress tracking for cloud runs | reconstructed after file loss; review before executing (references renumbered 2.6/2.7) |

## Archive

Shipped plans are in [`archive/`](archive/): 1.0 through 1.4, the 2.0 design
and implementation plans, the alternative-architecture proposal, and the
HuggingFace-backbone change overview.
