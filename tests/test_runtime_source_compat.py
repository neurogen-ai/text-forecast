from runtime.local import LocalRuntime


def test_local_runtime_supports_local() -> None:
    assert "local" in LocalRuntime.supported_source_backends


def test_local_runtime_does_not_support_modal() -> None:
    assert "modal" not in LocalRuntime.supported_source_backends
