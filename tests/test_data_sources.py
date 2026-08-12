from pathlib import Path

import pytest

from data.sources import (
    LocalDataSource,
    LocalSourceBackend,
    ModalDataSource,
    ModalSourceBackend,
    build_source_backend,
)


def test_local_data_source_resolve(tmp_path: Path) -> None:
    source = LocalDataSource(base_dir=tmp_path, name="foo")
    assert source.resolve() == tmp_path / "foo"
    assert source.backend == "local"


def test_local_source_backend(tmp_path: Path) -> None:
    backend = LocalSourceBackend({"base_dir": str(tmp_path)})
    source = backend.get_source("foo")
    assert isinstance(source, LocalDataSource)
    assert source.resolve() == tmp_path / "foo"


def test_local_source_backend_requires_existing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        LocalSourceBackend({"base_dir": str(missing)})


def test_modal_data_source_resolve() -> None:
    source = ModalDataSource(volume="my-volume", name="foo/bar")
    assert source.resolve() == Path("/modal/my-volume/foo/bar")
    assert source.backend == "modal"


def test_modal_source_backend() -> None:
    backend = ModalSourceBackend({"volume": "my-volume"})
    source = backend.get_source("foo/bar")
    assert isinstance(source, ModalDataSource)
    assert source.resolve() == Path("/modal/my-volume/foo/bar")


def test_build_source_backend_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown source backend"):
        build_source_backend("s3", {})
