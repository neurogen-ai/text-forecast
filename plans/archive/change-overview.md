# Change overview: swapping in a HuggingFace backbone classifier

This evaluates the codebase as it stands (post v1.0 dependency-injected experiments) and
spells out the smallest set of changes needed to train a model that loads GPT-2 (or any
`transformers` model), takes the final hidden embedding of one token, and feeds it through a
classification head that outputs `n_out` values.

## 1. How a run works today

An experiment module (`src/config/experiments/*.py`) is a self-contained `build()` function.
It receives a `RunContext`, an `Env`, and a `SourceBackend`, and returns an
`Experiment[T_Batch]`: model, strategy, tracker, two plain torch DataLoaders, and a
checkpoint processor. The CLI resolves the module by name (`config/loader.py`), and
`Engine.fit()` drives epochs: `strategy.training_step(batch)`, scheduler step, checkpoint,
eval epoch, metric report to MLflow.

The current text path is:

- `TextTokenDataset` scans staged parquet, concatenates token-list columns into `x`,
  thresholds `y` at `theta`, pads/truncates to `max_len`, and emits
  `(id, x, y, mask, weight)` as a NamedTuple. The mask is built from `x != pad_token_id`.
- `TransformerClass` embeds token ids itself (`nn.Embedding`), runs hand-written pre-norm
  RMSNorm blocks, then does attention-pooling over tokens via a learned scorer plus a linear
  head to `n_out`.
- `ClassificationStrategy` moves batches to device, calls `model.forward(batch)`, expects an
  output with `.logits` and `.probs`, computes BCE on probs, and hands logits/probs/y/ids to
  `BinaryClassificationTracker`.

## 2. Strengths of the current design

- **Experiment files are genuinely self-contained.** Everything about a run is visible in one
  file with no registry lookups or singletons. Adding a new model means writing one module;
  nothing else has to change. This is exactly the property that makes the swap below cheap.
- **Engine/Strategy split is clean.** The engine knows nothing about losses or optimizers;
  the strategy owns them. A new model only needs a strategy whose `_Batch` protocol matches.
- **Pydantic config schemas per model** catch bad hyperparameters before CUDA gets involved.
- **Checkpointing stores the experiment source file alongside weights**, so eval can rebuild
  the exact object graph later. A new model class rides along for free.
- **Datasets are decoupled from models.** `TextTokenDatasetOutput` already carries what a
  pretrained-backbone classifier needs: token ids, mask, target, id.

## 3. Stresses and issues found along the way

These matter because each one becomes a small tax when adding a HF-backed model.

1. **Model protocol drift.** `models/protocols.py::Model` declares a `config` class
   attribute. `TransformerClass` sets `config = ModelConfig`; older models like
   `TransformerLM` set `config_schema = ModelConfig` and a `filepath`. basedpyright strict
   apparently tolerates this because most old models don't declare conformance to the
   protocol. Any new code should follow `TransformerClass`, but the drift should be cleaned
   up eventually.
2. **Inconsistent forward signatures.** `TransformerClass.forward(batch)` takes a batch
   object; `TransformerLM.forward(x, mask)` takes tensors. The Strategy protocol assumes the
   former. Fine if you know it, annoying when it isn't documented anywhere central.
3. **Mask semantics are surprising.** `TextTokenDataset.__getitem__` builds
   `(x != pad).bool().unsqueeze(0).expand(max_len, -1)`, i.e. a square `(T, T)` mask where
   every row is identical. It works as an SDPA `attn_mask` broadcast, but a reader expects a
   key-padding mask of shape `(B, T)`. A HF backbone wants exactly `(B, T)` with 0 = pad, so
   this shape will need changing or adapting at the wrapper.
4. **Padding-token sentinel risk.** Masking by `x != pad_value` misclassifies any real token
   equal to the pad id. With GPT-2 there is no pad token at all; the usual convention
   (`pad_token_id = eos_token_id`) makes real EOS tokens invisible to the mask. This must be
   decided deliberately.
5. **Truncate-by-drop loses data silently.** `"truncate_method": "drop"` filters rows longer
   than `max_len` out of the dataset entirely rather than truncating them. For long
   abstracts under GPT-2's 1024 limit this could quietly discard much of the corpus. Worth a
   logged count at minimum.
6. **Whole-dataset materialisation in memory.** The dataset `collect()`s the filtered frame.
   Subsample paths collect too. Workable now, but a HF tokenizer applied lazily per-row would
   be preferable to re-running preprocessing whenever max_len changes.
7. **Loss on sigmoid probs.** `BinaryCrossEntropyLoss` uses `binary_cross_entropy` on
   `output.probs`. Numerically worse than `BCEWithLogitsLoss`, and it forces every model to
   expose a `.probs` field even when its head differs. A logits-only loss contract would be
   simpler and safer.
8. **Tracker coupling.** `BinaryClassificationTracker._log_plots` and its metric calcs assume
   a single logit/prob column (`squeeze(-1)` style). `n_out > 1` will crash or silently
   produce nonsense unless the tracker is generalised or a multi-label tracker is added.
9. **NaN-sentinel weight handling** (`weight = nan` meaning "unused") is clever but fragile;
   the loss checks `torch.any(torch.isnan(...))` per batch. An optional field would be more
   honest.
10. **Tokenizer assumptions live in comments.** `vocab_size=201_088` and
    `_PAD_TOKEN_ID = 0  # Must match the tokenizer used during preprocessing.` are implicit
    contracts between the preprocess app and the experiment file. Nothing validates them; a
    mismatch trains garbage without erroring.

## 4. Minimal changes for a GPT-2 backbone classifier

The good news: no changes to Engine, Strategy protocol, Experiment dataclass, loaders,
sources, or checkpointing. Four additions and three small edits.

### 4.1 New model: `src/models/gpt2_class.py`

```python
from typing import NamedTuple, Protocol

import torch
import torch.nn as nn
from pydantic import BaseModel, PositiveInt
from torch import Tensor
from transformers import AutoModel

from utils import component


class GPT2ClassifierConfig(BaseModel):
    model_name: str = "openai-community/gpt2"
    n_out: PositiveInt
    dropout: float = 0.1
    freeze_backbone: bool = False


class BatchInput(Protocol):
    x: Tensor      # (B, T) token ids
    mask: Tensor   # (B, T) bool/int, 1 = real token


class Output(NamedTuple):
    logits: Tensor
    probs: Tensor


@component
class GPT2Classifier(nn.Module):
    config = GPT2ClassifierConfig

    def __init__(self, config: GPT2ClassifierConfig, device, dtype):
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(config.model_name)
        if config.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        d = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(d, config.n_out),
        )

    def forward(self, batch: BatchInput) -> Output:
        # Accept either the dataset's (T, B)-expanded mask or a plain (B, T) mask.
        mask = batch.mask
        if mask.dim() == 3:
            mask = mask[0]
        out = self.backbone(input_ids=batch.x, attention_mask=mask.to(batch.x.dtype))
        h = out.last_hidden_state                       # (B, T, D)

        # Final non-pad token embedding per sequence.
        lengths = mask.sum(dim=1).clamp(min=1)          # (B,)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        last = h.gather(1, idx).squeeze(1)              # (B, D)

        logits = self.head(last).squeeze(-1)            # (B,) when n_out == 1
        return Output(logits=logits, probs=logits.sigmoid())
```

Design notes:

- "Final embedding layer" here means `last_hidden_state`; if you literally want the LM head
  tied embedding of the last token, substitute `backbone.wte(tokens_last)` for `last`. The
  hidden state is almost always the better feature.
- Gathering the last non-pad token beats reading position 0 or relying on a CLS token
  (GPT-2 has none), and it matches the causal-LM intuition that the final token has seen the
  whole prefix.
- Keeping the `Output(logits, probs)` NamedTuple means `ClassificationStrategy`,
  `BinaryCrossEntropyLoss`, and the tracker work unchanged for `n_out == 1`. That is the
  main reason this stays minimal.
- `AutoModel.from_pretrained` inside `__init__` means checkpoints store the full fine-tuned
  state dict (~500 MB for GPT-2 large). Acceptable given the existing MLflow artifact flow;
  note it in docs.

### 4.2 New experiment file: `src/config/experiments/gpt2_class.py`

Copy `transformer_class.py` and change: model construction (`GPT2Classifier` with
`n_out=1`), `_PAD_TOKEN_ID` (see 4.3), and optionally `max_len` up to 1024. Nothing else in
the file structure changes, which is the payoff of the v1.0 design.

### 4.3 Tokenizer / preprocessing edit

The staged parquet currently holds ids from whatever tokenizer preprocessing used
(vocab 201_088 suggests ModernBERT-family, not GPT-2's 50_257). Two options:

- **Minimal:** add a preprocess pass tokenising with `AutoTokenizer.from_pretrained("gpt2")`
  into a new staged dataset name (e.g. `all-lowercase-2-subset-gpt2`). Set
  `pad_token_id` to `tokenizer.eos_token_id` explicitly, since GPT-2 has no pad token. Be
  aware this makes real document-final EOS tokens indistinguishable from padding in the
  mask. If abstracts end with EOS, either append one synthetic EOS after truncation or mask
  on `x != pad | position == length-1`. The simplest robust fix: keep the dataset emitting
  the true lengths (or compute them from the mask) and let the model gather index
  `mask.sum(1) - 1`, as above.
- **Cleaner (later):** tokenize lazily in the dataset from raw text columns using a
  tokenizer named in the experiment config, eliminating the vocab-contract comment problem
  entirely. More invasive; not required for a first result.

Also fix `truncate_method`: with GPT-2's 1024 limit you want actual truncation
(`x[:max_len]`) rather than dropping rows, otherwise long papers vanish from training.

### 4.4 Small edits

- `TextTokenDataset.__getitem__`: emit the mask as plain `(B, T)`
  (`(x != pad).bool()`) instead of the expanded square. The only consumer is the model, and
  both the new wrapper and `TransformerClass`'s SDPA call can be adjusted in the same commit
  (`attn_mask=mask[:, None, None, :]` style broadcast). Doing this now removes issue 3
  permanently.
- `models/__init__.py`: export `GPT2Classifier` (the build-helper regenerates this).
- Optionally `losses/binary_cross_entropy.py`: switch to
  `binary_cross_entropy_with_logits(output.logits, ...)` and drop the sigmoid round-trip.
  One-line change, better numerics, and it removes the requirement that future heads expose
  `.probs`.

### 4.5 What deliberately stays untouched

- `Engine`, `Experiment`, `Strategy` protocol: the new model fits the existing
  `forward(batch) -> (logits, probs)` contract for `n_out=1`.
- Checkpoint processors: they save arbitrary `state_dict`s plus the experiment file.
- Trackers: fine for binary `n_out=1`. Generalising to `n_out > 1` (multi-label BCE with
  per-class metrics, or softmax CE for multiclass) is a second task: a new loss module plus a
  multi-label tracker, selected from a new experiment file. No shared code needs to change,
  which is the right test of the abstractions, and they pass it.

## 5. Summary of effort

| Change | Size |
|---|---|
| `models/gpt2_class.py` wrapper | new file, ~60 lines |
| `experiments/gpt2_class.py` | copy + ~10 line diff |
| Tokenizer preprocess pass (GPT-2 vocab, explicit pad id) | small addition to preprocess config/run |
| Dataset mask shape `(B, T)` | few lines, one consumer to update |
| Loss to logits-based | one line |

Total: roughly a day including a smoke run. The architecture earns its keep here: the
experiment-module design means the change is additive, and the only genuinely risky part is
the pad/EOS masking semantics, which deserves a unit test on the gather-index logic before
any GPU time.

## 6. Recommended abstraction amendments

The changes above get a GPT-2 model trained with almost no surgery, but they leave several
of the frictions in section 3 in place. These are the fixes I'd make, roughly in priority
order.

### 6.1 Pad token: yes, just hard-code it per backbone class

You're right that this doesn't need an elaborate abstraction. The pad id is a property of
the tokenizer, and the wrapper class already knows which tokenizer it loads. So put it there:

```python
class GPT2Classifier(nn.Module):
    PAD_TOKEN_ID = 50256  # eos; GPT-2 has no dedicated pad token

    def __init__(self, config, device, dtype):
        ...
        tok = AutoTokenizer.from_pretrained(config.model_name)
        self.pad_token_id = tok.pad_token_id or tok.eos_token_id
```

and have the experiment file read `model.pad_token_id` when constructing the dataset config,
instead of hand-copying `_PAD_TOKEN_ID = 0` between files. That single indirection kills the
vocab/pad contract drift (issue 10) for this class without introducing any new protocol.
A `HFBackboneClassifier` base could later do this once for every `transformers` model, but I
wouldn't build that until there's a second backbone to justify it. One concrete class per
backbone, each owning its own pad id and pooling choice, is simpler and more honest than a
premature generic layer.

The EOS-as-padding ambiguity remains even with the right id, so keep the gather-last-token
logic plus its unit test regardless.

### 6.2 Fix the batch contract, then enforce it

The biggest structural weakness is that "what a model may expect from a batch" exists only as
scattered Protocols (`BatchInput`, `BCEBatch`, `_Batch` in the strategy) that don't reference
each other. Amend:

- Move one canonical batch type (or a small family: text-token, graph, scalar) into
  `data/datasets/types.py` and have datasets, models, strategies, and losses all import it.
- Make the mask `(B, T)` key-padding shape part of that definition, and delete the
  expand-to-square hack in `TextTokenDataset`. One commit fixes issue 3 everywhere.
- Optionally add a tiny conformance test: every class decorated `@component` under
  `src/models/` must accept a synthetic batch of the canonical type. This turns the silent
  forward-signature drift (issue 2) into a loud test failure.

### 6.3 Standardise the Model protocol

Collapse `config` / `config_schema` / `filepath` into one form following
`TransformerClass`: a `config: type[BaseModel]` class attribute and
`forward(batch) -> Output`. Migrate old models mechanically or mark legacy ones as such.
Until this is done, basedpyright strict gives false confidence about model compatibility.

### 6.4 Change the loss contract to logits-only

Replace the implicit "model must expose `.probs`" requirement with losses that consume
logits:

- Binary: `binary_cross_entropy_with_logits(output.logits, target)`.
- Multi-class / multi-label: plain cross-entropy over `n_out` dims.

Then `Output.probs` becomes optional (computed only where a metric needs it), models stop
doing sigmoid inside `forward`, and numerics improve for free. This is the single highest
value-to-effort amendment on the list.

### 6.5 Make truncation explicit and counted

In `TextTokenDatasetConfig`, replace the free-text `truncate_method: str` with a
`Literal["truncate", "drop"]`, log how many rows each path affects, and default to
`"truncate"`. Dropping data should be something you opt into and see numbers for.

### 6.6 Tracker decoupling

Trackers currently infer everything from store-name prefixes and assume column shapes.
Minimum viable fix: give `MetricTracker.calc_metrics` an explicit `n_out` (derivable from
the first logged logits tensor's last dim) and branch binary vs multi-label there, rather
than adding a new tracker subclass per output arity. The existing prefix scheme can stay;
only the squeeze logic needs arity awareness.

### 6.7 Replace NaN sentinels with Optional

`weight: Tensor | None` in the batch type, `weights: ... | None` already exists in config.
The loss checks `if batch.weight is not None`. Removes the per-batch `torch.any(isnan)`
scan and the trap of a legitimate NaN weight meaning "unused".

### What not to do

- Don't introduce a generic `BackboneClassifier` hierarchy yet; two concrete classes
  (`TransformerClass`, `GPT2Classifier`) don't justify it, and the experiment-file pattern
  means duplication costs almost nothing.
- Don't route tokenisation into the dataset lazily until a second run actually needs it;
  the staged-parquet approach is fine while preprocessing is cheap relative to training.

Net effect: 6.1 through 6.4 are each under a day, they remove four of the ten issues in
section 3 outright and turn the rest into test failures instead of silent misbehaviour, and
none of them change the Engine/Experiment/Strategy contracts that are working well.
