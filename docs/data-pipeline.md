# Data pipeline

`src/data/` covers everything between raw parquet files and model-ready
batches: sources (where data lives), the preprocess and engineer pipelines
(batch ETL), datasets and samplers (per-example serving), and formaters
(per-row value transforms).

## Sources (`data/sources/`)

Two protocols define the interface:

- `DataSource` - a handle that resolves to a path readable by the current
  process. Local files resolve to `base_dir / name`; Modal volumes resolve to
  `/modal/<volume>/<name>` inside a container.
- `SourceBackend` - a named factory for data sources, built from a config dict.

Backends register themselves via `@register_source_backend("local" | "modal")`.
The CLI selects one with `--source-backend`; see docs/apps.md for the flags.
Adding a backend means implementing both classes and registering them.

## Preprocess pipeline (`data/preprocess/`)

The pipeline is described by an ordered tuple of step specs (plan 2.1):

- `CleanStep` - one text column, with flags for lowercasing, quality-marker
  nulling, language policy, length trim, terminal-period requirement
  (`steps.py`)
- `DropNaStep` - drop rows null in any of the given columns
- `TokeniseStep` - tokeniser key + columns
- `EmbedStep` - embedder key + constructor kwargs + columns

Each spec is a frozen kw-only dataclass with a `tag` literal for
serialisation (`step_to_dict` / `step_from_dict` dispatch on the tag). The
CLI parses ordered `--op` flags into this tuple (`apps/preprocess_ops.py`);
see docs/apps.md for the flag surface.

`PreprocessJob` is a frozen, serialisable description of a job: global filter
and partition parameters plus the `steps` tuple. It carries no runtime state;
`Runtime.run_preprocess` decides whether to run it locally or dispatch it to a
remote container (the Modal runtime pickles whole jobs via `fn.remote(job)`, so
step specs need only be picklable, but dict serialisation is kept for metadata
and tests).

`run_preprocess_pipeline` scans all parquet files under the origin source,
applies global filters up front:

- license filter (`is_license_safe`, on by default)
- publication date window (`start_date` inclusive, `end_date` exclusive)
- field id, language, and document type whitelists

It then assigns rows to partitions (`--partitions` or `--rows-per-part`;
defaults to one partition per input file) and processes each partition in
sequence, applying license scrubbing first (columns in
`replace_non_permissive_cols` are nulled where `is_license_safe` is false),
then each pipeline step in order via its registered handler (`HANDLERS` in
`pipeline.py`), sinking to `part_<i>.parquet` with zstd compression. Step
handlers:

- **CleanStep** (`clean.py`) - see below.
- **DropNaStep** - `lf.drop_nulls(subset=cols)`.
- **EmbedStep** - embed columns are wrapped in BOS/EOS tokens, concatenated,
  and encoded via `map_batches`. Output column: `<cols>_embedding` as a
  fixed-width Float32 array.
- **TokeniseStep** - `<col>_tokens` as Int64 lists, nulls become empty lists.

A `metadata.json` of all job params (including the serialised steps) is written
alongside the parts. `--dry-run` slices 500 rows and prints instead of writing.

### CleanStep semantics (`clean.py`)

Applied to a single column, in order: lowercase (optional) → quality markers →
language handling → length trim → terminal period. Rows whose text contains any
`EXCLUDE_QUALITY` substring (boilerplate like "log in", "uses cookies", "an
abstract is not available") are nulled unless `drop_quality=False`. Rows
containing a non-English marker from `EXCLUDE_LANG` (CJK function words,
accented characters) follow `lang_policy`: `mark` sets `language` to
`"unknown"` without dropping (default), `drop` removes them, `off` ignores
them.

Length trim (`LengthTrim`) semantics were corrected in 2.1:

- `min_chars` is a hard minimum: rows with `len_chars < min_chars` are dropped.
- When `max_sigma` is set, mean/std of character length are computed once per
  column per partition (one streaming collect) and rows with
  `len > mean + max_sigma * std` are dropped.
- There is deliberately **no lower statistical bound**: short rows are governed
  solely by `min_chars`.

#### Old level → equivalent CLI flags

Pre-2.1 cleaning used cumulative integer levels (`--clean-level N`). The new
surface composes the same effects from explicit flags. Note that old level 3's
lower bound was `max(mean - 3σ, min_len)` applied through two comparisons that
had their signs inverted, so in practice the old code dropped the *wrong* rows
(it nulled long in-band rows on one side and kept extreme outliers on the
other). The mapping below describes intended old behaviour; actual old
behaviour differed wherever the sign bug bit. Level 3+ behaviour intentionally
changes beyond the sign fix: short-row handling is now purely `min_chars`, with
no statistical floor.

| Old level | Equivalent flags |
|-----------|------------------|
| 1 | `--op "clean:<col>"` alone (quality filter + language marking) |
| 2 | `--op "clean:<col>" --lowercase` |
| 3 | `--op "clean:<col>" --trim-min-chars <min_len> --trim-max-sigma 3` (intended behaviour; see sign-bug caveat above) |
| 4 | same as 3 (`max_sigma 3` gives both sides' intent; there is no separate lower σ cut anymore) |
| 5 | `--op "clean:<col>" --require-terminal-period` plus the desired trim flags |

Tokenise/embed/dropna steps map directly: `--op "tokenise:<cols>"
--op-tokeniser whitespace`, `--op "embed:<cols>" --op-embedder modernbert-base`,
`--op "dropna:<cols>"`.

### Embedders (`embed.py`, `embed_huggingface.py`, `embed_modal.py`)

Embedders implement the `TextEmbedder` protocol: `output_dim` plus
`encode(texts) -> list[list[float]]`, registered under a string key with
`@register_embedder`.

Currently registered keys (defined in `EMBEDDERS`):

- `modernbert-base` - answerdotai/ModernBERT-base, 768-dim
- `modernbert-embed-base` - nomic-ai/modernbert-embed-base, 768-dim

Both use `HuggingFaceEmbedder`: lazy loading, mean pooling, configurable dtype
(bfloat16 default), device, batch size, optional torch.compile. Empty texts are
replaced with the unk token before encoding. The registry also carries each
key's BOS/EOS tokens so the pipeline can wrap inputs consistently.

On the modal runtime, `runtime.get_embedder` may return the Modal GPU-backed
implementation instead; the pipeline code is identical either way.

### Tokeniser (`tokenise.py`)

Wraps a HuggingFace `AutoTokenizer` (`use_fast=False`). On first use it
downloads and saves the tokeniser locally; subsequent partitions reuse the
loaded instance. Adds BOS/EOS special tokens, no truncation or padding.

## Engineer pipeline (`data/pipeline/engineer.py`)

Same job-object pattern as preprocess (`EngineerJob` -> `run_engineer_pipeline`,
dispatched by `Runtime.run_engineer`). Reads the origin dataset, splits into 64
partitions by default, and computes:

- **Column lengths** - `<col>_len` for each `len_cols` entry (string or list
  columns only).
- **Years to first citation** - two paths:
  - from the paper's own `counts_by_year` array: `counts_by_year_delta` (years
    minus publication year) and its first element;
  - from the citation graph: explode `referenced_works`, invert to get citing
    dates per cited id, join back, then derive `cited_by_delta_years_first`
    and `cited_by_delta_days_first`.

The years-to-first path requires `publication_date` as a date type and
list-typed `referenced_works` and `counts_by_year_years`. Older staged datasets
that store these as string/binary columns fail here and need re-staging.

The file also contains `run_dbscan_on_chunk`, which clusters per-paper citation
distributions with DBSCAN (eps=15, weighted by counts) to find citation-count
clusters, centroids, member sizes, and noise ratio. It is not called by the
main pipeline loop yet.

Output is zstd parquet parts plus `metadata.json` recording the job parameters
and the runtime that produced the dataset (`local` or `modal`).

## Describe (`data/pipeline/describe.py`)

`DescribeJob` backs the `describe` CLI app: time-window filtering on a chosen
column, optional Polars filter expression, per-column statistics with optional
bucket boundaries. Dispatched through `Runtime.run_describe`.

## Datasets (`data/datasets/`)

Datasets are PyTorch `Dataset`s configured by Pydantic models and instantiated
through the `@component` factory pattern. Common config knobs across datasets:
column selection (`x`, `y`), padding/truncation (`max_len`, `pad_token_id`),
temporal windowing (`time_col`, `t_start`, `t_end`, used by eval's sliding
windows), sampling limits (`sample`, `subsample`), and id passthrough.

Base classes:

- **PolarsDataset** (`polars_dataset.py`) - base for tabular serving from
  parquet. Handles loading, filtering, shuffling, padding/masking, weights,
  and target distribution plotting. Returns `(id, x, y, weight, mask)`.

Task-specific variants override `_format_y` mostly:

- **BinaryThresholdDataset** - binarises a continuous target at `theta`
  (inclusive).
- **BinaryCategorialDataset** - binary labels from categorical columns.
- **OrdinalDataset** - bins a continuous target into ordinal classes via
  `boundaries`; also returns `y_orig`.
- **LogRegressDataset** - log-transformed regression target.
- **TextTokenDataset** - concatenates requested token columns (e.g.
  `title abstract_tokens`) for plain sequence classification. No graph
  neighbourhood. Suits TransformerClass-style models.
- **GenerativePretrainDataset(2)** - language-modelling style datasets for
  pretraining.

Graph datasets build citation-graph neighbourhoods per example:

- **CitationGraphDataset** (`citation_graph_dataset.py`) - the full-featured
  graph dataset: neighbour lookup, top-k selection, category/sort columns.
- **GraphDataset** (`graph_dataset.py`) - generic base with `graph_max_len`,
  `top_k`, `add_x`; supports `with_window(t_start, t_end)` temporal slicing,
  which eval depends on.
- **OrdinalGraphDataset**, **BinaryThresholdGraphDataset** - task heads on top
  of graph serving.

## Samplers (`data/samplers/`)

- **AutoBinnedSampler** - a `WeightedRandomSampler` that bins the target column
  (from `df_y` or an attribute), computes inverse-frequency bin weights, and
  samples uniformly across bins. For heavily skewed citation targets. Runs the
  target through the dataset's `_format_y` first, so it matches what the loss
  will see.
- **PortionSampler** - draws a fixed number of random indices without weights;
  cheap subsampling.

## Formaters (`data/formaters/`)

A `Formater` is a per-row callable that transforms already-selected values into
model tensors. It deliberately does not choose columns; datasets own column
selection, formaters own value formatting.

- **GraphFormater** - tensor conversion, x truncation, graph_x pad/truncate to
  `top_k`, class-weight lookup.
- **CitationGraphFormater** - flattening and padding to `max_len` /
  `graph_max_len` with the pad token id, plus mask construction.

Formaters are frozen dataclasses, so they compose cleanly with the Pydantic
dataset configs.
