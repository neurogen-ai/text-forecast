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
| [2.2](2.2.md) | Modal runtime tidy-up: dedupe config/log noise, inline wrappers, single-source EVAL naming | drafted; [implementation plan](implementation/v2.2/README.md) written (branches A–F), not started |
| [2.3](2.3.md) | Production deployment hardening | not started |
| [2.4](2.4.md) | Distributed GPU training/eval | not started |
| [2.5](2.5.md) | Remote progress tracking for cloud runs | reconstructed after file loss; review before executing |

## Archive

Shipped plans are in [`archive/`](archive/): 1.0 through 1.4, the 2.0 design
and implementation plans, the alternative-architecture proposal, and the
HuggingFace-backbone change overview.
