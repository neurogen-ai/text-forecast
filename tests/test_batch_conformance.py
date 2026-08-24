"""Conformance test: every migrated ``@component`` model must accept a
synthetic canonical batch (plan 1.4.1) and return a value exposing
``.logits``.

Models on the skip list are excluded deliberately, with the reason recorded
here so exclusion stays visible (plan 1.4.2 migrates or retires them).
"""

from pathlib import Path

import pytest
import torch
from torch import Tensor

from data.datasets.types import TokenBatch
from models import TransformerClass, TransformerLM
from models.transformerClass import ModelConfig
from models.transformerLM import ModelConfig as LMModelConfig

B = 4
T = 8
VOCAB = 64


def synthetic_token_batch() -> TokenBatch:
    generator = torch.Generator().manual_seed(0)
    x = torch.randint(0, VOCAB, (B, T), generator=generator)
    # Left-pad half of each row so the key-padding mask is exercised.
    lengths = torch.randint(T // 2, T + 1, (B,), generator=generator)
    mask = torch.arange(T)[None, :] < lengths[:, None]
    x = x * mask  # pad positions become id 0
    return TokenBatch(
        id=torch.arange(B),
        x=x,
        y=torch.randint(0, 2, (B, 1), generator=generator).float(),
        mask=mask,
        weight=None,
    )


def check_logits(model: object, out: object, n_out: int) -> None:
    logits = getattr(out, "logits", None)
    assert isinstance(logits, Tensor), f"expected .logits tensor, got {type(out)}"
    assert logits.shape == (B, n_out), f"bad logits shape {tuple(logits.shape)}"


def test_transformer_class_conforms() -> None:
    config = ModelConfig(
        n_heads=2,
        n_layers=2,
        vocab_size=VOCAB,
        embed_dim=16,
        hidden_dim=32,
        n_out=1,
        dropout=0.01,
    )
    model = TransformerClass(config, torch.device("cpu"), torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.forward(synthetic_token_batch())
    check_logits(model, out, config.n_out)


def test_transformer_lm_conforms() -> None:
    config = LMModelConfig(
        model_name="test-lm",
        pad_token_id=0,
        n_heads=2,
        n_layers=1,
        vocab_size=VOCAB,
        embed_dim=16,
        hidden_dim=32,
        n_out=VOCAB,
        dropout=0.01,
    )
    model = TransformerLM(config, torch.device("cpu"), torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.forward(synthetic_token_batch())
    logits = getattr(out, "logits", None)
    assert isinstance(logits, Tensor), f"expected .logits tensor, got {type(out)}"
    # LM head emits per-position logits: (B, T, n_out).
    assert logits.shape == (B, T, config.n_out), f"bad logits shape {tuple(logits.shape)}"


def test_transformer_lm_generate_smoke() -> None:
    config = LMModelConfig(
        model_name="test-lm",
        pad_token_id=0,
        n_heads=2,
        n_layers=1,
        vocab_size=VOCAB,
        embed_dim=16,
        hidden_dim=32,
        n_out=VOCAB,
        dropout=0.01,
    )
    model = TransformerLM(config, torch.device("cpu"), torch.float32)
    model.eval()
    with torch.no_grad():
        tokens = model.generate(synthetic_token_batch().x[:, :4], max_len=3)
    # generate() appends one token per batch row per step.
    assert len(tokens) == 3 * 4


MIGRATED = {"transformerClass.py", "transformerLM.py"}
NON_MODEL_MODULES = {"__init__.py", "protocols.py"}


def test_every_model_is_migrated_or_visibly_legacy() -> None:
    """Each module under src/models/ must either be covered by a conformance
    test here or carry an explicit ``# legacy:`` header (plan 1.4.2). Exclusion
    from the contract must be visible, never accidental."""
    models_dir = Path(__file__).parents[1] / "src" / "models"
    unaccounted = []
    for path in sorted(models_dir.glob("*.py")):
        if path.name in NON_MODEL_MODULES or path.name in MIGRATED:
            continue
        header = path.read_text()[:600]
        if not header.startswith("# legacy:"):
            unaccounted.append(path.name)
    assert not unaccounted, (
        f"models missing protocol migration or '# legacy:' header: {unaccounted}"
    )
