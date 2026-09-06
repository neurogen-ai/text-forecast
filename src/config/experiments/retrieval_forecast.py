"""Retrieval-forecast experiment.

Task: predict whether a paper ends up with more than 5 citations, using the
paper's own title/abstract tokens plus neighbours retrieved from a vector
database of abstract embeddings.

Corpus filters:

- Vector database: all papers with ``cited_by_count >= 1`` (a paper with at
  least one citation has proven impact worth retrieving).
- Training/eval rows: the same ``>= 1`` filter; the binary target is the
  ``> 5 citations`` threshold (``theta=5.0``), so the model separates
  moderate from high-impact papers rather than cited from uncited.

Pipeline: ``AbstractQueryEmbedder`` produces ``N`` query embeddings per
paper; each query first retrieves a pool of ``n_candidates`` rows from the
database (FAISS when installed, chunked matmul otherwise) under ``no_grad``,
then a differentiable top-``top_k`` selection runs over that pool with the
straight-through estimator from CLaRa (He et al., 2026, Algorithm 1): the
forward pass consumes exactly the hard top-k picks, the backward pass uses
the softmax gradient over the masked pool, so gradients reach the embedder
through the selection rather than a soft weighting. The ``N * top_k``
selected vectors cross-attend with the target token sequence inside
``RetrievalForecast``.

Recency filter: the vector store loads ``publication_date`` for every db row
(``delta_years=1.0``), and each example's batch date (``return_date=True``)
restricts search to rows published at least one year before the example, so
no paper retrieves neighbours from its own future (or itself).
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import date

import polars as pl
import torch
from torch.utils.data import DataLoader

from config.env import Env
from config.experiment import Experiment
from config.runtime import RunContext
from data.datasets.retrieval_dataset import (
    RetrievalDataset,
    VectorStoreDataset,
    VectorStoreDatasetConfig,
)
from data.datasets.text_token_dataset import (
    TextTokenDatasetConfig,
    token_batch_collate,
)
from data.datasets.types import TokenBatch
from data.sources import build_default_source_backend
from models.retrieval_forecast import (
    AbstractQueryEmbedder,
    RetrievalForecast,
    RetrievalModelConfig,
)
from training.checkpointing import MlflowCheckpointProcessor
from training.losses import BinaryCrossEntropyLoss
from training.optimizers.specs import AdamWSpec
from training.schedulers import WarmupCosineSpec
from training.strategies import ClassificationStrategy, StrategyConfig
from training.tracking import BinaryClassificationTracker

experiment_name: str = "RetrievalForecast-cited-gr-1-theta-5"

_SOURCE_NAME = "all-lowercase-2-embedded"
_X_COLS = ["title_tokens", "abstract_tokens"]
_Y_COL = ["citation_normalized_percentile"]
_MAX_LEN = 256
_PAD_TOKEN_ID = 0  # Must match the tokenizer used during preprocessing.
_BATCH_SIZE = 8
_NUM_WORKERS = 2
_EPOCHS = 10

# Retrieval hyperparameters.
_N_QUERIES = 64  # N: query embeddings per paper
_TOP_K = 2  # vectors selected per query -> retrieved sequence length 32
_N_CANDIDATES = 50_000  # stage-1 pool size for the CLaRa ST top-k (paper: 20)
_DB_MAX_ROWS = 250_000  # cap on corpus vectors loaded into the database
_EMBEDDING_COL = "abstract_embedding"  # precomputed parquet embedding column
_EMBEDDING_DIM = 768  # output dim of the registered preprocess embedders
_DELTA_YEARS = 1.0  # retrieved rows must be >= this many years older

# Binary target: cited_by_count > 5, over papers with >= 1 citation.
_THETA = 0.75


def build(
    runtime: RunContext,
    env: Env,
) -> Experiment[TokenBatch]:
    """Build the retrieval-forecast experiment."""
    device = runtime.device
    dtype = runtime.dtype
    subsample = runtime.subsample

    source = build_default_source_backend(env).get_source(_SOURCE_NAME)

    # Both training rows and the retrieval corpus keep only cited papers.
    filter_expr = ((pl.col('cited_by_count') >= 1) & (pl.col('field_name') == 'Chemistry'))

    base_dataset_kwargs = {
        "loc": _SOURCE_NAME,
        "x": _X_COLS,
        "y": _Y_COL,
        "meta_cols": ['field_name', 'cited_by_count'],
        "filter": filter_expr,
        "weights": torch.tensor([0.816, 1.291]),
        "max_len": _MAX_LEN,
        "pad_token_id": _PAD_TOKEN_ID,
        "pad": True,
        "truncate": True,
        "truncate_method": "drop",
        "auto_remove": True,
        "time_col": "publication_date",
        "id_col": "id",
        "return_id": True,
        "return_date": True,
        "theta": _THETA,
    }

    train_config = TextTokenDatasetConfig(
        **base_dataset_kwargs,
        name="train-retrieval",
        t_start=date(1950, 1, 1),
        t_end=date(2015, 1, 1),
        subsample=subsample,
    )
    val_config = TextTokenDatasetConfig(
        **base_dataset_kwargs,
        name="val-retrieval",
        t_start=date(2015, 1, 1),
        t_end=date(2018, 1, 1),
        subsample=subsample,
    )
    # Vector database: precomputed embeddings of all papers with >= 1
    # citation, loaded straight from the parquet embedding column.
    store_config = VectorStoreDatasetConfig(
        loc=_SOURCE_NAME,
        embedding_col=_EMBEDDING_COL,
        filter=filter_expr,
        max_rows=_DB_MAX_ROWS,
        normalize=True,
        delta_years=_DELTA_YEARS,
        name="vector-store",
    )
    store = VectorStoreDataset(config=store_config, source=source)

    train_dataset = RetrievalDataset(config=train_config, source=source)
    val_dataset = RetrievalDataset(config=val_config, source=source)

    model_config = RetrievalModelConfig(
        n_heads=4,
        n_layers=8,
        vocab_size=201_088,
        embed_dim=_EMBEDDING_DIM,  # must match the precomputed embedding dim
        hidden_dim=2048,
        n_out=1,
        dropout=0.1,
        n_queries=_N_QUERIES,
        top_k=_TOP_K,
        n_candidates=_N_CANDIDATES,
        max_len=_MAX_LEN,
    )
    embedder = AbstractQueryEmbedder(
        vocab_size=model_config.vocab_size,
        embed_dim=model_config.embed_dim,
        n_queries=model_config.n_queries,
        n_heads=model_config.n_heads,
        dropout=model_config.dropout,
        device=device,
        dtype=dtype,
    )
    model = RetrievalForecast(
        config=model_config,
        embedder=embedder,
        db=store.db,
        store_search=store.search,
        device=device,
        dtype=dtype,
        store_dates=store.dates,
        delta_years=store.config.delta_years,
    )

    loss_fn = BinaryCrossEntropyLoss(config=None)
    tracker = BinaryClassificationTracker(device=device, dtype=dtype)

    if device.type == "cuda":
        stream = torch.cuda.Stream()
        stream_context = torch.cuda.stream(stream)
    else:
        stream_context = nullcontext()

    optimizer_spec = AdamWSpec(lr=1e-4, weight_decay=1e-3)
    scheduler_spec = WarmupCosineSpec(
        milestones=(2,),
        warmup_start_factor=1e-5,
        eta_min=1e-6,
        epochs=_EPOCHS,
    )

    strategy = ClassificationStrategy(
        config=StrategyConfig(
            model=model,
            loss_fn=loss_fn,
            tracker=tracker,
            optimizer_spec=optimizer_spec,
            scheduler_spec=scheduler_spec,
            stream=stream_context,
            device=device,
            accumulation_steps=1,
            examples_per_epoch=len(train_dataset),
            mat_mul_precision="high",
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=_BATCH_SIZE,
        num_workers=_NUM_WORKERS,
        prefetch_factor=4,
        persistent_workers=False,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        collate_fn=token_batch_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=_BATCH_SIZE,
        num_workers=_NUM_WORKERS,
        prefetch_factor=4,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        collate_fn=token_batch_collate,
    )

    checkpoints = MlflowCheckpointProcessor(
        artifact_loc=env.artifact_loc,
        tracking_uri=env.tracking_uri,
        experiment_name=experiment_name,
    )

    return Experiment(
        experiment_name=experiment_name,
        model=model,
        strategy=strategy,
        tracker=tracker,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoints=checkpoints,
        epochs=_EPOCHS,
        eval_interval=1,
        checkpoint_interval=2,
    )
