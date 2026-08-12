from pathlib import Path

from apps.source_args import build_source_backend_from_cli
from config.env import Env
from data.sources import LocalDataSource, ModalDataSource


def test_build_source_backend_defaults_to_local(env: Env, tmp_path: Path) -> None:
    env.source["local"]["base_dir"] = str(tmp_path)
    backend = build_source_backend_from_cli(
        env=env,
        source_backend=None,
        source_opts=[],
        source_base_dir=None,
        source_volume=None,
    )
    assert backend.name == "local"
    source = backend.get_source("foo")
    assert isinstance(source, LocalDataSource)
    assert source.resolve() == tmp_path / "foo"


def test_build_source_backend_selects_modal(env: Env) -> None:
    backend = build_source_backend_from_cli(
        env=env,
        source_backend="modal",
        source_opts=[],
        source_base_dir=None,
        source_volume=None,
    )
    assert backend.name == "modal"
    source = backend.get_source("foo")
    assert isinstance(source, ModalDataSource)
    assert source.resolve() == Path("/modal/test-volume/foo")


def test_source_base_dir_shortcut(env: Env, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    backend = build_source_backend_from_cli(
        env=env,
        source_backend=None,
        source_opts=[],
        source_base_dir=other,
        source_volume=None,
    )
    assert backend.name == "local"
    assert backend.get_source("foo").resolve() == other / "foo"


def test_source_volume_shortcut(env: Env) -> None:
    backend = build_source_backend_from_cli(
        env=env,
        source_backend=None,
        source_opts=[],
        source_base_dir=None,
        source_volume="other-volume",
    )
    assert backend.name == "modal"
    assert backend.get_source("foo").resolve() == Path("/modal/other-volume/foo")


def test_source_opt_override(env: Env, tmp_path: Path) -> None:
    env.source["local"]["base_dir"] = str(tmp_path)
    backend = build_source_backend_from_cli(
        env=env,
        source_backend=None,
        source_opts=["base_dir=/tmp/other"],
        source_base_dir=None,
        source_volume=None,
    )
    assert backend.get_source("foo").resolve() == Path("/tmp/other/foo")
