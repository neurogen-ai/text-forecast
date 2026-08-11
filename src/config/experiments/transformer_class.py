"""Basic Transformer classifier experiment.

Inputs: tokenised title/abstract (``title abstract_tokens``).
Target: binary ``cited_by_count > 0`` (one or more citations).
Model: ``TransformerClass``.

Assumes the staged parquet dataset contains a tokenised list column named
``title abstract_tokens`` (produced by the preprocess tokenise step) and a
scalar target column ``cited_by_count``.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import date
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader

from config.env import Env
from config.experiment import Experiment
from config.runtime import RunContext
from data.datasets.text_token_dataset import TextTokenDataset, TextTokenDatasetConfig
from data.datasets.text_token_dataset import TextTokenDatasetOutput
from data.sources import LocalStagedSource
from models import TransformerClass
from training.checkpointing import MlflowCheckpointProcessor
from training.losses import BinaryCrossEntropyLoss
from training.optimizers.specs import AdamWSpec
from training.schedulers import WarmupCosineSpec
from training.strategies import ClassificationStrategy, StrategyConfig
from training.tracking import BinaryClassificationTracker

experiment_name: str = "Basic-TransformerClass-cited-gr-1"


# Sensible defaults for a small binary classifier.
_SOURCE_NAME = "all-lowercase-2-subset"
_X_COLS = ["title_tokens", "abstract_tokens"]
_Y_COL = ["cited_by_count"]
_MAX_LEN = 256
_PAD_TOKEN_ID = 0  # Must match the tokenizer used during preprocessing.
_BATCH_SIZE = 32
_NUM_WORKERS = 2
_EPOCHS = 8


def build(runtime: RunContext, env: Env) -> Experiment[TextTokenDatasetOutput]:
    """Build the basic Transformer classifier experiment."""
    device = runtime.device
    dtype = runtime.dtype
    subsample = runtime.subsample

    source = LocalStagedSource(
        path=env.staged_loc,
        name=_SOURCE_NAME,
    )

    # Keep only rows with a non-null citation count.
    filter_expr = pl.col(_Y_COL) >= 0

    base_dataset_kwargs = {
        "loc": _SOURCE_NAME,
        "x": _X_COLS,
        "y": _Y_COL,
        "meta_cols": [],
        "filter": filter_expr,
        "weights": None,
        "max_len": _MAX_LEN,
        "pad_token_id": _PAD_TOKEN_ID,
        "pad": True,
        "truncate": True,
        "truncate_method": "drop",
        "auto_remove": True,
        "time_col": "publication_date",
        "id_col": "id",
        "return_id": True,
        "theta": 0.0,  # cited_by_count > 0
    }

    train_config = TextTokenDatasetConfig(
        **base_dataset_kwargs,
        name="train-text-tokens",
        t_start=date(1920, 1, 1),
        t_end=date(1990, 1, 1),
        subsample=subsample,
    )
    val_config = TextTokenDatasetConfig(
        **base_dataset_kwargs,
        name="val-text-tokens",
        t_start=date(1990, 1, 1),
        t_end=date(1991, 1, 1),
        subsample=subsample,
    )

    train_dataset = TextTokenDataset(config=train_config, source=source)
    val_dataset = TextTokenDataset(config=val_config, source=source)

    model_config = TransformerClass.config(
        n_heads=4,
        n_layers=8,
        vocab_size=201_088,
        embed_dim=256,
        hidden_dim=1024,
        n_out=1,
        dropout=0.1,
    )
    model = TransformerClass(config=model_config, device=device, dtype=dtype)

    loss_fn = BinaryCrossEntropyLoss(config=None)
    tracker = BinaryClassificationTracker(device=device, dtype=dtype)

    if device.type == "cuda":
        stream = torch.cuda.Stream()
        stream_context = torch.cuda.stream(stream)
    else:
        stream_context = nullcontext()

    optimizer_spec = AdamWSpec(lr=1e-3, weight_decay=1e-3)
    scheduler_spec = WarmupCosineSpec(
        milestones=(0,),
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
        shuffle=False,
        drop_last=True,
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
