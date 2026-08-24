# Citation Forecast

A CLI-driven ML pipeline that forecasts academic impact from paper metadata.
Built as a hands-on project for experimenting with NLP forecasting and deep
learning engineering.

## What it does

The pipeline takes academic paper data (text, citations, graph structure),
preprocesses and embeds it, then trains models to predict future citation
counts. Experiments are tracked with MLflow; inference can be served on Modal's
serverless GPUs.

## How a run works

1. Pick an experiment module with `--experiment` (or set it in `config/config.toml`).
   Experiment modules live in `src/config/experiments/` and declare the full
   object graph for a run: model, datasets, samplers, DataLoader, loss,
   strategy, tracker, and checkpoint processor, all validated by Pydantic.
2. The CLI (`src/main.py`) resolves the experiment and dispatches to a Typer subcommand:
   - `preprocess` - clean, tokenise, embed the raw data
   - `describe` - dataset statistics
   - `engineer` - feature engineering pipeline
   - `eval` - config-driven evaluation, temporal scrubbing, JSON export for graphs
   - `train` - build everything from the experiment module and hand off to the generic Engine
3. The Engine (`src/training/engine.py`) runs the epoch/batch loop, calling into
   trackers, checkpointing backends, and callbacks.

## Layout

- `src/apps/` - Typer CLI apps (train, eval, preprocess, describe, engineer)
- `src/builders/` - factory helpers for progress bars and run components
- `src/config/` - Pydantic schemas, experiment modules, env/runtime config, loader
- `src/data/`
  - `datasets/` - custom PyTorch datasets (token, ordinal, binary, graph variants)
  - `sources/` - pluggable data sources (local filesystem, Modal volume)
  - `samplers/` - binned and portion-based samplers for imbalanced targets
  - `preprocess/` - cleaning, tokenisation, HuggingFace/Modal embedding
  - `pipeline/`, `formaters/` - feature engineering and formatting
- `src/models/` - model implementations: transformer classifiers/LMs, MLPs,
  RNNs, graph models, heteroscedastic regression heads
- `src/training/` - the Engine plus losses, optimizers, schedulers, strategies,
  metric trackers, and checkpointing (local, MLflow store, S3)
- `src/runtime/` - local vs Modal execution backends
- `src/utils/` - logging, root-dir resolution, parquet export, registries
- `tests/` - test suite

## Tech stack

- Python 3.13, PyTorch
- MLflow for experiment tracking, Modal for serverless GPU inference
- Polars for loading (predicate pushdown) and multi-worker serving
- Pydantic for config validation
- Typer + Rich for the CLI

## Docs

- `docs/apps.md` - the CLI apps: train, eval, preprocess, describe, engineer
- `docs/training.md` - the Engine loop, strategies, losses, optimizers/schedulers, trackers
- `docs/data-pipeline.md` - sources, preprocessing, datasets, samplers, formaters
- `docs/experiments.md` - writing experiment modules: contract, `Experiment` dataclass, worked examples

## Further reading

- `README.md` - feature highlights and version-by-version roadmap
- `plans/` - per-release planning notes and change overviews
