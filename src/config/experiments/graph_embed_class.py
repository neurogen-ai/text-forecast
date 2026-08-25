"""First dependency-injected experiment file.

This is a 1:1 port of the historical defaults from the old singleton config
module (``config/data.py``, ``config/model.py``, ``config/train.py``) into a
single self-contained experiment module per plan 1.0 (v3) phase 4.
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
from data.datasets.graph_dataset import GraphDataset, GraphDatasetConfig
from data.datasets.types import CitationGraphDatasetOutput
from data.formaters import GraphFormater
from data.sources import build_default_source_backend
from models import EmbedGraphClass
from training.checkpointing import MlflowCheckpointProcessor
from training.losses import BinaryCrossEntropyLoss
from training.optimizers.specs import AdamWSpec
from training.schedulers import WarmupCosineSpec
from training.strategies import ClassificationStrategy, StrategyConfig
from training.tracking import BinaryClassificationTracker

experiment_name: str = "General-2-graph-embed"


def build(
    runtime: RunContext,
    env: Env,
) -> Experiment[CitationGraphDatasetOutput]:
    """Build the full experiment object graph from the injected runtime and env."""
    device = runtime.device
    dtype = runtime.dtype
    subsample = runtime.subsample

    source = build_default_source_backend(env).get_source(
        "1920-2000-lowercase-2-embedded"
    )

    filter_expr = (pl.col("cited_by_count") >= 1) & (
        pl.col("referenced_works").list.len() >= 5
    )

    base_dataset_kwargs = {
        "x": ["title abstract_embedding"],
        "y": ["cited_by_delta_years_first"],
        "meta_cols": ["field_id", "subfield_name", "cited_by_count"],
        "filter": filter_expr,
        "pad_value": torch.zeros(768, dtype=torch.float32),
        "shuffle": False,
        "sample": None,
        "loc": "1920-2000-lowercase-2-embedded",
        "return_mask": True,
        "truncate": True,
        "truncate_method": "drop",
        "pad": True,
        "auto_remove": True,
        "time_col": "publication_date",
        "category_cols": ["topic_name", "subfield_name", "field_name", "type"],
        "sort_cols": ["publication_date"],
        "top_k": 15,
        "add_x": ["title abstract_embedding"],
        "graph_max_len": 100,
        "max_mem_rows": int(1e6),
    }

    train_config = GraphDatasetConfig(
        **base_dataset_kwargs,
        name="train-dataset",
        max_len=400,
        t_start=date(1920, 1, 1),
        t_end=date(1990, 1, 1),
        return_id=False,
        weights=torch.tensor([1.07, 0.93]),
        subsample=subsample,
    )
    val_config = GraphDatasetConfig(
        **base_dataset_kwargs,
        name="test-dataset",
        max_len=400,
        t_start=date(1990, 1, 1),
        t_end=date(1991, 1, 1),
        return_id=True,
        weights=None,
        subsample=subsample,
    )

    train_formater = GraphFormater(
        max_len=400,
        graph_max_len=100,
        top_k=15,
        pad=True,
        truncate=True,
        truncate_method="drop",
        return_mask=True,
        return_id=False,
        weights=torch.tensor([1.07, 0.93]),
    )
    val_formater = GraphFormater(
        max_len=400,
        graph_max_len=100,
        top_k=15,
        pad=True,
        truncate=True,
        truncate_method="drop",
        return_mask=True,
        return_id=True,
        weights=None,
    )

    train_dataset = GraphDataset(
        config=train_config,
        source=source,
        formater=train_formater,
    )
    val_dataset = GraphDataset(
        config=val_config,
        source=source,
        formater=val_formater,
    )

    model_config = EmbedGraphClass.config(
        vocab_size=201_088,
        dtype=torch.float32,
        n_heads=4,
        input_seq_len=150,
        latent_seq_len=1000,
        n_layers=24,
        embed_dim=768,
        hidden_dim=768 * 4,
        n_out=1,
        causal_mask=False,
        dropout=0.05,
    )
    model = EmbedGraphClass(config=model_config, device=device, dtype=dtype)

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
        epochs=8,
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
        batch_size=16,
        num_workers=2,
        prefetch_factor=4,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        num_workers=2,
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
        epochs=8,
        eval_interval=1,
        checkpoint_interval=2,
    )
