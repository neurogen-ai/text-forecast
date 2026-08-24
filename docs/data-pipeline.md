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

`PreprocessJob` is a frozen, serialisable description of a job. It carries no
runtime state; `Runtime.run_preprocess` decides whether to run it locally or
dispatch it to a remote container.

`run_preprocess_pipeline` scans all parquet files under the origin source,
applies global filters up front:

- license filter (`is_license_safe`, on by default)
- publication date window (`start_date` inclusive, `end_date` exclusive)
- field id, language, and document type whitelists

It then assigns rows to partitions (`--partitions` or `--rows-per-part`;
defaults to one partition per input file) and processes each partition in
sequence, sinking to `part_<i>.parquet` with zstd compression. Per partition,
in order:

1. **License scrubbing** - columns in `replace_non_permissive_cols` are nulled
   where `is_license_safe` is false.
2. **Cleaning** - `clean.py` per column/level/min_len triple.
3. **Null drops** - per `drop_na_cols`.
4. **Embedding** - if an embedder key is set, embed columns are wrapped in BOS/EOS
   tokens, concatenated, and encoded via `map_batches`. Output column:
   `<cols>_embedding` as a fixed-width Float32 array.
5. **Tokenising** - `<col>_tokens` as Int64 lists, nulls become empty lists.

A `metadata.json` of all CLI params is written alongside the parts. `--dry-run`
slices 500 rows and prints instead of writing.

### Cleaning levels (`clean.py`)

Cleaning is cumulative; level N includes everything below:

| Level | Effect |
|-------|--------|
| 1 | Null out rows whose text contains any `exclude_quality` marker (boilerplate like "log in", "uses cookies", "an abstract is not available") |
| 2 | Lowercase the column |
| 3 | Null out rows below `mean - 3*std` length (floor at `min_len`) |
| 4 | Also null rows above `mean + 3*std` |
| 5 | Keep only rows ending in a period |

Rows containing any non-English marker from `exclude_lang` (CJK function words,
accented characters) have their `language` set to `"unknown"` rather than being
dropped.

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
