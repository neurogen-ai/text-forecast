# Experiments

An experiment module is a single self-contained Python file in
`src/config/experiments/` that declares the full object graph for a run:
datasets, model, loss, optimizer, scheduler, strategy, tracker, DataLoaders,
and checkpoint processor. There is no hidden global config; everything a run
needs is built in one place, so `train`, `eval`, and MLflow all see the same
setup.

## Contract

A valid experiment module must define two module-level names (enforced by
`config.loader.load_experiment`):

- `experiment_name: str` - the MLflow experiment name
- `build(runtime, env) -> Experiment` - constructs and returns
  the object graph

The file name (without `.py`) is what you pass to `--experiment` or set as
`experiment` in `config/config.toml`. Resolution order: CLI flag > config.toml >
error listing available experiments.

## The Experiment dataclass

Defined in `config/experiment.py`:

| Field | Purpose |
| --- | --- |
| `experiment_name` | MLflow experiment name |
| `model` | `nn.Module`; must expose `.compile(mode=...)` for the `train --compile` flag |
| `strategy` | owns the training step: loss, optimizer, scheduler, grad accumulation |
| `tracker` | computes/logs epoch metrics |
| `train_loader`, `val_loader` | plain torch DataLoaders over your datasets |
| `checkpoints` | `CheckpointProcessor` (MLflow store, local, or S3) |
| `epochs`, `eval_interval`, `checkpoint_interval` | loop cadence |

Runtime concerns stay out of the module where possible. `build()` receives:

- `RunContext` - `device`, `dtype`, `compile_mode`, `fullgraph`,
  `subsample` (from `--subsample`; thread it into dataset configs so dry runs
  actually shrink)
- `Env` - tracking URI, artifact location, and source config

Dataset location is machine-level and arrives through `env.source`. Call
`build_default_source_backend(env).get_source(name)` to resolve a staged
dataset by name; the dataset *name* itself stays in the experiment file.

## Writing an experiment

Start from `transformer_class.py` or `graph_embed_class.py`. The usual shape:

1. **Module docstring** stating inputs, target, and any assumptions about the
   staged data (column names, tokeniser pad id).
2. **Constants** for source name, columns, hyperparameters.
3. **Datasets.** Build train/val configs with disjoint time windows
   (`t_start`/`t_end` on `publication_date`). Pass `subsample=runtime.subsample`
   through so `--dry-run` style testing works.
4. **Model** via its `.config(...)` constructor, then instantiate with
   `runtime.device` and `runtime.dtype`.
5. **Loss + tracker**, matched to task type (e.g.
   `BinaryCrossEntropyLoss` + `BinaryClassificationTracker`).
6. **Strategy.** `ClassificationStrategy` (or regression variants) wrapping a
   `StrategyConfig`: model, loss, tracker, `AdamWSpec`, scheduler spec, CUDA
   stream context (use `nullcontext()` on CPU), accumulation steps,
   `examples_per_epoch=len(train_dataset)`.
7. **DataLoaders.** Plain `torch.utils.data.DataLoader`s.
8. **Checkpoint processor.** Typically
   `MlflowCheckpointProcessor(artifact_loc=env.artifact_loc,
   tracking_uri=env.tracking_uri, experiment_name=experiment_name)`.
9. Return the `Experiment`.

### Example skeleton

```python
experiment_name: str = "My-experiment-v1"

def build(runtime, env):
    device, dtype = runtime.device, runtime.dtype
    subsample = runtime.subsample

    source = build_default_source_backend(env).get_source(_SOURCE_NAME)
    train_ds = TextTokenDataset(config=train_config, source=source)  # t_end 1990
    val_ds = TextTokenDataset(config=val_config, source=source)      # 1990-1991

    model = MyModel(config=MyModel.config(...), device=device, dtype=dtype)
    loss_fn = BinaryCrossEntropyLoss(config=None)
    tracker = BinaryClassificationTracker(device=device, dtype=dtype)

    stream_context = (
        torch.cuda.stream(torch.cuda.Stream())
        if device.type == "cuda" else nullcontext()
    )
    strategy = ClassificationStrategy(config=StrategyConfig(
        model=model, loss_fn=loss_fn, tracker=tracker,
        optimizer_spec=AdamWSpec(lr=1e-3, weight_decay=1e-3),
        scheduler_spec=WarmupCosineSpec(milestones=(0,), ...),
        stream=stream_context, device=device,
        accumulation_steps=1,
        examples_per_epoch=len(train_ds),
        mat_mul_precision="high",
    ))

    return Experiment(
        experiment_name=experiment_name, model=model, strategy=strategy,
        tracker=tracker,
        train_loader=DataLoader(train_ds, ...),
        val_loader=DataLoader(val_ds, ...),
        checkpoints=MlflowCheckpointProcessor(
            artifact_loc=env.artifact_loc,
            tracking_uri=env.tracking_uri,
            experiment_name=experiment_name,
        ),
        epochs=8, eval_interval=1, checkpoint_interval=2,
    )
```

## Existing examples

- `transformer_class.py` - binary classifier ("cited at least once") over
  tokenised title/abstract text using `TextTokenDataset` and `TransformerClass`.
- `graph_embed_class.py` - same target over embedded text plus citation-graph
  neighbours, using `GraphDataset` + `GraphFormater` and `EmbedGraphClass`.
  Shows class weights (`weights=torch.tensor([1.07, 0.93])`), filters on list
  columns, and meta/category columns.

## Eval and experiment files

When you train, the app saves the experiment module itself to the checkpoint
store (see `_experiment_file_path` in `apps/train.py`). `eval --run-id ...`
downloads that file and loads it with `load_experiment_from_path`, rebuilding
the exact configuration used at training time. Consequence: after changing an
experiment module mid-project, old runs still evaluate correctly against their
original config, but new fields read by old files must keep working. Keep
modules backward compatible or accept that eval of old runs uses their frozen
copy.

## Checklist for a new experiment

- [ ] File in `src/config/experiments/`, unique stem, defines `experiment_name` and `build()`
- [ ] Train/val windows don't overlap; val window matches how you plan to use eval
- [ ] `runtime.subsample` passed into dataset configs
- [ ] Pad token id / embedding dims match the preprocessed source
- [ ] Tracker matches loss and task type
- [ ] `examples_per_epoch` set from the actual train dataset length
