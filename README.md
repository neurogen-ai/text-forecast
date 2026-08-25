# Text Forecast

Welcome to text-forecast, a CLI-based machine learning framework for
forecasting outcomes from text corpora. It began life as a citation-impact
predictor, but nothing in the pipeline is specific to academic papers:
embed a text dataset, track it across time windows, train, and evaluate.
I built this as a hands-on environment to experiment with NLP forecasting,
while extending my practical knowledge of 
Python, software engineering, and deep learning.

The core training loop and data pre-processing is driven entirely by
CLI and integrates directly with MLflow to track experiments and compare
iterations iterations efficiently. Since v2.0 it is a hybrid local/cloud system:
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

* **Data:** Polars (predicate pushdown loading), Polars (multi-worker serving)

* **Validation:** Pydantic (Type safe model config schemas)

* **Tooling:** Typer (CLI), Rich (Console logging)

# 2. Road-map

## Completed

<details> 
<summary> <b>Expand List</b></summary> 

> ## v0.2
> <details>
> <summary><b>v0.2.0 Data pre-processing CLI</b></summary>
>
> - Standardise data pre-processing notebook into separate functions.
> - Orchestrate pre-processing functions in main loop.
> - CLI control via Typer app, functions induced/skipped via option flags.
>
> </details>


> ## v0.3
> <details>
> <summary><b>v0.3.0 Dataloader Flexibility</b></summary>
> 
> - Support concatenation of multiple string/token columns when serving examples from dataloader.
> 
> </details>


> ## v0.4 
>
> <details>
> <summary><b>v0.4.0 Eval CLI</b></summary>
> 
> - CLI app for evaluating models
> - Config driven with optional CLI overrides 
> - Temporal scrub with variable intervals and MLflow tracking
> - JSON data export for website historical performance graphs
> 
> </details>
> 
> <details>
> <summary><b>v0.4.1 Metric Tracker Generalisation</b></summary>
> 
> - Generalise & simplify metric tracker methods/initialisation
> - Generalised parameter init function & metric calc functions responsive to store name prefix
> 
> </details>
> 
> 
> <details>
> <summary><b>v0.4.2 Standardise cross-app arg parsing</b></summary>
> 
>- Standardise start/end date arg parsing across apps
>
> </details>
>
>
> <details>
> <summary><b>v0.4.3 Update data loading & handling</b></summary>
>
> - Specialised dataset load funcs, more efficient mem use
> - Ignore extra cols (e.g index) during parquet data load
> - Add no. of examples in dataset to train/eval metric/param logging 
>
> </details>
>
> <details>
> <summary><b>v0.4.4 Enhanced Metric calculation</b></summary>
>
> - Add 'best threshold' metric calculation for accuracy, also recording precicion/recall at that threshold
> - PR & ROC AUC charts logged as mlflow artifacts during evaluation step
> 
> </details>
>
> <details>
> <summary><b>v0.4.5 Preprocess &amp Train app upgrade</b></summary>
>
> - Add metadata export, drop count logging, more user warnings, to pre-processing 
> - Cleaning has more granular control with column specific 'levels', tidied argument names
> - Cleaning split into 'drop' and 'clear' (replace with null)
> - License based filtering and clearing added to clean 
> - Tokenise step leaves empty list for nulls instead of filling nulls with empty string prior to tokenisation
> - Pre-processing sped up by partitioning via filter instead of slicing
> - Log model file as mlflow arifact 
>
> </details>


> ## 0.5 
> <details>
> <summary><b>v0.5.0 Dataset ipc refactor </b></summary> 
>
> - Moved from in mem df to ipc cached dataset post filtering operations for fast random access of OOM dataset rows
> - Now organise dataset & their cache under the 'name' attribute 
>
> </details>
>
> <details>
> <summary><b>v0.5.1 Descriptives app </b></summary> 
>
> - Added descriptives app, allowing CLI driven descriptives generation of specific columns across datasets
> - Shows Polars descriptives table and relative/total frequency counts for variable bucket boundaries
> - Calculates proportional weights for balanced training with n_buckets classes 
>
> </details>
> 
> 
> <details>
> <summary><b>v0.5.2 Metric tracker efficiency &amp Model train checkpoint loading</b></summary>
> 
> - Removed redundant recall/precision calculations from best accuracy metric calc
> - Added model checkpoint to be loaded in train loop
> 
> </details>
> 
> <details>
> <summary><b>v0.5.3 Efficiency, Control &amp Clean update </b></summary>
> 
> - More efficient best accuracy calculations across a smaller range & no. of values 
> - Train loop can load mlflow stored checkpoints & set parent run via CLI args
> - Dataset formatting classes available via registry in CLI / config
> - Dataset kwargs specified in CLI / config for flexible dataset initialisation
> - Polars max threads, and compile mode determined by CLI args
> - Learning rate scheduler args determined by config 
> - New lowercase clean method for string columns
> 
> </details>
>
> ## v0.6
> <details>
> <summary><b>v0.6.0 Model Eval Metric export for visualisation</b></summary>
>
> - Dataset outputs dataclass for resilient batch access to optional fields with dot operator access
> - Associate model outputs with input row id to categorise and measure correlation of metrics
> - Structured JSON metric exports for modularised records
>
> </details>
>
> ## v0.7
> <details>
> <summary><b>v0.7.0 CLI &amp Config Consolidation</b></summary>
>
> - Add config/env value overrides to train app
> - Move code-as-config module from root to src, allow config value overrides from CLI via option flags
> - Refactor train (main) loop into /apps
> - Create metric tracker base class for tracking logic, overide metric calculation in children 
> - Seperate train/data/env configs into distinct files
> - Add dedicated loss/optimisation config for lr schedule milestones etc.
>
> </details>
>
> ## v1.0
> <details>
> <summary><b>v1.0.0 Dependency-injected experiment configs</b></summary>
>
> - One self-contained experiment module per experiment under `src/config/experiments/` declares the full object graph
> - Generic `Experiment[T_Batch]` dataclass holds model, strategy, tracker, plain torch DataLoaders, and checkpoint processor
> - `Engine` owns the epoch/batch loop; `Strategy` implements `training_step`, `validation_step`, and `configure_optimizers`
> - Typed optimizer/scheduler specs (`AdamWSpec`, `WarmupCosineSpec`) built inside the experiment file
> - Checkpoint processor abstraction (local, MLflow, S3 stub) saves full dict checkpoints and the original experiment file as a run artifact
> - `train` and `eval` apps rewired to load experiments; `eval` downloads the run's experiment file and rebuilds its windows
> - Tracker rewrite: dedicated subclasses, dict CPU stores, explicit kwargs, no singleton config reads
> - Runtime `Registry` replaced by `@component` marker + `utils.build_helper` for package `__init__` blocks
> - basedpyright strict, Python 3.13, keyword-only constructors throughout
>
> </details>

> ## v1.1
> <details>
> <summary><b>v1.1.0 Migrate remaining apps + robust preprocessing + local/Modal prep</b></summary>
>
> - Migrate `preprocess`, `describe`, and `engineer` apps to load machine settings via `config.env.load_env(...)`
> - Add `[env]` CLI override flags to all remaining apps
> - Remove the temporary Phase-0 `config/env.py` `__getattr__` shim
> - Make preprocessing robust: lazy embedder loading, pluggable embedding models, CPU/CUDA support
> - Introduce `Runtime` and `DataSource` abstractions so local and Modal execution share the same app code
>
> </details>

> ## v1.2
> <details>
> <summary><b>v1.2.0 Local / Modal runtime split</b></summary>
>
> - Implement `modal` runtime backend using the abstractions from v1.1
> - Add Modal volume-backed data sources and Modal GPU embedder
> - Run `preprocess` with `--runtime modal` (train/eval stay local; see v2.0)
>
> </details>

> ## v1.3
> <details>
> <summary><b>v1.3.0 Modal runtime for describe and engineer</b></summary>
>
> - Runtime-agnostic `DescribeJob` / `EngineerJob` pipelines under `src/data/pipeline/`, moved out of the apps
> - `Runtime` protocol extended with `get_source`, `run_describe`, `run_engineer`; local and Modal backends implement it
> - `describe` and `engineer` dispatch through the runtime with `--runtime modal`; no app-side runtime branching
> - Describe and engineer run as whole CPU jobs in the shared `text-forecast-data` Modal app against the staged volume
> - Engineer writes `metadata.json` recording the runtime that produced the dataset
>
> </details>

> ## v2.0
> <details>
> <summary><b>v2.0.0 Training and evaluation on Modal</b></summary>
>
> - Experiments build from `(runtime, env)` alone; machine locations live in `[source]` config, never CLI flags
> - Venue-independent `TrainJob` / `EvalJob` pipelines under `src/training/pipeline/`; apps are thin parse-and-delegate layers
> - MLflow runs created client-side by the runtime wrapper and resumed in-place by `run_id`, so run naming/parenting works the same on both venues
> - `ModalTrainingGPU` spawns fire-and-forget train/eval jobs with `.spawn()`; checkpoints land as MLflow artifacts, so a Modal-trained run resumes locally and vice versa
> - Eval prediction exports are `exports/<year>` MLflow artifacts (downloadable from anywhere), not local directories
> - Engine and trackers run headless (`progress=None`) inside containers
>
> </details>

</details>

# 3. Project Structure
```text
text-forecast/
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
└── text-forecast                       # CLI entry point (pyproject.toml script)
```

# 4. Quick start
* Clone the repository and install the package. The `text-forecast` command is created automatically from the `pyproject.toml` console-script entry point.
```bash
git clone https://github.com/Felix-Noble/text-forecast.git
cd text-forecast
pip install .
```

> For development, install in editable mode with `pip install -e .`.

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

* Run the MLflow tracking server
```bash
# activate venv containing mlflow, or use uv/pipx
mlflow server
```

* Start a training run on CPU using the experiment selected in `config.toml`:
```bash
text-forecast train -s smoke --no-gpu --subsample 512
```

* Or select an experiment explicitly on the CLI:
```bash
text-forecast --experiment graph_embed_class train -s smoke --no-gpu --subsample 512
```

* Evaluate a checkpoint over sliding one-year windows (exports land on the
  eval run as `exports/<year>` MLflow artifacts):
```bash
text-forecast eval -id <run-id> -e <epoch> -s 1990-01-01 -i 1 --dry-run --no-gpu
```

* Resume any run from its MLflow checkpoint — including runs trained on Modal:
```bash
text-forecast train -s resume --load-id <run-id> --load-epoch <n> --no-gpu
```

### On Modal

Install the optional extra and fill in the `[runtime.modal]` section of
`config/config.toml` (project, volumes, GPUs, timeout). Then preprocess,
train, and evaluate on Modal GPUs with one flag:
```bash
pip install '.[modal]'

text-forecast preprocess --runtime modal          # stage data onto the Modal volume
text-forecast --experiment graph_embed_class train -s smoke --subsample 512 --runtime modal
text-forecast eval -id <modal-run-id> -e <epoch> -s 1990-01-01 -i 1 --dry-run --runtime modal
```
Train jobs are spawned fire-and-forget: the CLI prints the MLflow run id and
Modal FunctionCall id and returns immediately. Block until done with
`modal FunctionCall.from_id(<id>).get()`, or watch the run in the MLflow UI.
Checkpoints stream into the same run either way, so cross-runtime resume needs
no extra steps.
