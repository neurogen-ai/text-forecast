from pathlib import Path

from config.env import Env, _deep_update


def test_deep_update_replaces_nested_values() -> None:
    base: dict = {"local": {"base_dir": "/tmp/data"}}
    _deep_update(base, {"local": {"base_dir": "/new/data"}})
    assert base["local"]["base_dir"] == "/new/data"


def test_deep_update_adds_keys() -> None:
    base: dict = {"local": {"base_dir": "/tmp/data"}}
    _deep_update(base, {"modal": {"volume": "v"}})
    assert base["modal"]["volume"] == "v"
    assert base["local"]["base_dir"] == "/tmp/data"


def test_env_source_defaults() -> None:
    env = Env(
        tracking_uri="http://example.com",
        artifact_loc=Path("/tmp/artifacts"),
        source={"default": "modal", "modal": {"volume": "v"}},
    )
    assert env.source["default"] == "modal"
