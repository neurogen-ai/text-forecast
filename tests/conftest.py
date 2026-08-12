from pathlib import Path

import pytest

from config.env import Env


@pytest.fixture
def env(tmp_path: Path) -> Env:
    """Return an Env with source backends pointing at temp directories."""
    return Env(
        tracking_uri="http://127.0.0.1:5000",
        artifact_loc=tmp_path / "artifacts",
        runtime={"default": "local"},
        source={
            "default": "local",
            "local": {"base_dir": str(tmp_path / "data")},
            "modal": {"volume": "test-volume"},
        },
    )
