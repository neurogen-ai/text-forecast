# Forecite

Welcome to forecite, a CLI-based machine learning framework for
forecasting outcomes from text corpora. It began life as a citation-impact
predictor, but nothing in the pipeline is specific to academic papers:
embed a text dataset, track it across time windows, train, and evaluate.
I built this as a hands-on environment to experiment with NLP forecasting,
while extending my practical knowledge of 
Python, software engineering, and deep learning.

The core training loop and data pre-processing is driven entirely by
CLI and integrates directly with MLflow to track experiments and compare
iterations efficiently. Since v2.0 it is a hybrid local/cloud system:
each experiment is a single self-contained Python module that instantiates its
own models, datasets, samplers, plain torch DataLoaders, strategy, tracker, and
checkpoint processor, and every app runs either locally or on Modal GPUs with
one `--runtime` flag.

# 1. Tech Stack & Key Features
## 1.1 Key Features
* **Modular Experiment Architecture:**
Self-contained experiment modules under `src/config/experiments/` declare the full object graph for a run; the CLI selects the experiment and delegates the epoch/batch loop to a generic Engine.

* **Local or cloud execution:**
One `Runtime` interface dispatches every job. Train and evaluate locally, or
spawn headless GPU jobs on Modal (`--runtime modal`) against volume-backed
data, with checkpoints and metrics flowing into the same MLflow runs either way.

* **Data Pipeline:** 
Data pre-processing CLI and custom PyTorch Datasets/Loaders forming a flexible ETL pipeline

## 1.2 Tech Stack
* **Core:** Python (3.13), PyTorch (Deep Learning)

* **MLOps:** MLflow (tracking), Modal (Deployment), Scikit-Learn (metric calculation)

* **Data:** Polars (predicate pushdown loading, multi-worker serving)

* **Validation:** Pydantic (Type safe model config schemas)

* **Tooling:** Typer (CLI), Rich (Console logging)

# 2. Road-map

The full version-by-version breakdown lives in [ROADMAP.md](ROADMAP.md), and
shipped releases are itemised in [CHANGELOG.md](CHANGELOG.md). In brief:

* **Shipped (v0.2 → v2.4.4):** evolved from a pre-processing/dataloader
  prototype through the v1.x experiment-architecture and runtime-abstraction
  releases into the current hybrid system — v2.0 brought training and
  evaluation on Modal GPUs, v2.1 rebuilt the preprocessing pipeline on
  ordered step specs, v2.3 added the retrieval-forecast model and vector
  store dataset, and the v2.4 line expanded query search (candidate
  strategies, corpus scope, compute dtypes, Gumbel sampling) plus
  preprocess embed/tokenise fixes.
* **Active next** (per [`plans/releases/README.md`](plans/releases/README.md)):
  2.5 Modal runtime tidy-up and preprocess resume, 2.6 production hardening,
  2.7 distributed training, 2.8 remote progress tracking, 2.9 CLI launch
  speed (lazy heavy imports) — all drafted or not yet started.

# 3. Project Structure
```text
forecite/
├── config/
│   └── config.toml             # experiment name + [env] machine settings
├── plans/                      # implementation plans
├── production/
│   ├── service.py              # modal inference image
│   └── models/                 # config, architecture, and weights for production models
│       └── examp-model/
│           ├── model/
│           │   ├── arch.py     # architecture and config schema
│           │   └── config.py   # hyperparameter config
│           ├── tokeniser/      # tokeniser (transformers)
│           └── weights/        # model checkpoint (.pt)
├── src/
│   ├── main.py                 # root Typer app: --experiment/-e selection
│   ├── apps/                   # typer CLI argument parsing (train, eval, preprocess, describe, engineer)
│   ├── builders/               # app-side chrome (progress bars)
│   ├── config/
│   │   ├── experiments/        # one self-contained experiment module per experiment
│   │   ├── loader.py           # experiment module resolution/loading
│   │   ├── runtime.py          # RunContext (device/dtype/compile/subsample)
│   │   └── env.py              # Env dataclass loaded from config.toml
│   ├── runtime/                # execution backends for local/Modal jobs
│   │   ├── base.py             # Runtime + TextEmbedder protocols
│   │   ├── local.py            # LocalRuntime
│   │   ├── modal_runtime.py    # volumes, GPU embedder, remote job dispatch
│   │   └── factory.py          # build_runtime()
│   ├── data/
│   │   ├── datasets/           # PyTorch datasets
│   │   ├── formaters/          # per-row value transforms
│   │   ├── samplers/           # samplers for PyTorch Datasets
│   │   ├── sources/            # DataSource protocol + local/modal backends
│   │   ├── preprocess/         # clean/tokenise and stage dataset selections
│   │   └── pipeline/           # runtime-agnostic describe/engineer jobs
│   ├── models/                 # PyTorch modules with Pydantic config schemas
│   └── training/
│       ├── engine.py           # owns the epoch/batch loop
│       ├── strategies/         # train/val step logic + configure_optimizers
│       ├── optimizers/         # optimizer specs
│       ├── schedulers.py       # LR scheduler specs
│       ├── losses/             # loss functions
│       ├── tracking/           # metric tracking, calculation, and MLflow logging
│       ├── checkpointing/      # local/MLflow/S3 checkpoint processors
│       └── callbacks/          # early stopping
├── utils/
│   ├── registry.py             # @component marker decorator
│   └── build_helper.py         # regenerates package __init__.py auto blocks
└── forecite                       # CLI entry point (pyproject.toml script)
```

# 4. Quick start
* Clone the repository and install the package with a torch extra. Torch is
**not** a core dependency — it arrives via one of the mutually exclusive
`cpu` / `cuda126` / `cuda130` extras (`pyproject.toml` wires these to the
matching PyTorch wheel indexes for uv). The `forecite` command is
created automatically from the console-script entry point.

**CUDA (NVIDIA GPU):**
```bash
git clone https://github.com/Felix-Noble/forecite.git
cd forecite
uv sync --extra cuda126   # CUDA 12.6 wheels; use --extra cuda130 for CUDA 13.0
uv run forecite train -s smoke --gpu --subsample 512
```

**CPU-only:**
```bash
git clone https://github.com/Felix-Noble/forecite.git
cd forecite
uv sync --extra cpu
uv run forecite train -s smoke --no-gpu --subsample 512
```

**pip equivalents** (pip ignores `[tool.uv.sources]`, so point it at the
PyTorch index explicitly):
```bash
# CUDA 12.6
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install '.[cuda126]'
# CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install '.[cpu]'
```

> For development, `uv sync --extra cpu` (or your GPU extra of choice)
> installs the project editable; add the `dev` dependency group for pytest
> (`uv sync --extra cpu --group dev`).

* Configure `config/config.toml` with your machine settings:
```toml
experiment = "graph_embed_class"

[env]
tracking_uri = "http://127.0.0.1:5000"
artifact_loc = "/path/to/experiment-tracking/artifacts"

[source]
default = "local"

[source.local]
base_dir = "/path/to/staged/data"

[runtime]
default = "local"
```

* Run the MLflow tracking server (mlflow is installed with the package, so
the project venv already has it):
```bash
mlflow server
```

* Start a training run on CPU using the experiment selected in `config.toml`:
```bash
forecite train -s smoke --no-gpu --subsample 512
```

* Or select an experiment explicitly on the CLI:
```bash
forecite --experiment graph_embed_class train -s smoke --no-gpu --subsample 512
```

* Evaluate a checkpoint over sliding one-year windows (exports land on the
  eval run as `exports/<year>` MLflow artifacts):
```bash
forecite eval -id <run-id> -e <epoch> -s 1990-01-01 -i 1 --dry-run --no-gpu
```

* Resume any run from its MLflow checkpoint — including runs trained on Modal:
```bash
forecite train -s resume --load-id <run-id> --load-epoch <n> --no-gpu
```

### On Modal

Install the optional extra (alongside a torch extra) and fill in the
`[runtime.modal]` section of `config/config.toml` (project, volumes, GPUs,
timeout — see `config/config.example.toml`). Then preprocess, train, and
evaluate on Modal GPUs with one flag:
```bash
# one-time: authenticate Modal locally
modal token new

pip install '.[modal]'   # or: uv sync --extra cuda126 --extra modal

# deploy the Modal app (required for preprocess/describe/engineer — they
# dispatch to the deployed app by name; train/eval spawn ephemeral jobs)
modal deploy src/runtime/modal_runtime.py

forecite preprocess --runtime modal --source-backend modal   # stage data onto the Modal volume
forecite --experiment graph_embed_class train -s smoke --subsample 512 --runtime modal
forecite eval -id <modal-run-id> -e <epoch> -s 1990-01-01 -i 1 --dry-run --runtime modal
```
The modal runtime only supports the modal *source* backend, so preprocess
must pass `--source-backend modal` (or use `--source-volume <name>`) unless
`[source].default` in `config.toml` is already `"modal"`.
Re-deploy after changing `src/` or `config/` — the deploy bakes the current
source tree and config into the container image, and preprocess/describe/
engineer dispatch to the deployed app by name (`forecite-data`).
Train jobs are spawned fire-and-forget: the CLI prints the MLflow run id and
Modal FunctionCall id and returns immediately. Block until done with
`modal FunctionCall.from_id(<id>).get()`, or watch the run in the MLflow UI.
Checkpoints stream into the same run either way, so cross-runtime resume needs
no extra steps.
