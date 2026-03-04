from __future__ import annotations

from litestar_saq.cli import _resolve_multiprocessing_context as resolve_cli_context
from litestar_saq.plugin import SAQPlugin


class _DummyContext:
    def __init__(self, start_method: str) -> None:
        self._start_method = start_method

    def get_start_method(self) -> str:
        return self._start_method


class _DummyMultiprocessing:
    def __init__(self, default_method: str = "forkserver", fork_available: bool = True) -> None:
        self.default_context = _DummyContext(default_method)
        self.fork_context = _DummyContext("fork")
        self.spawn_context = _DummyContext("spawn")
        self.fork_available = fork_available

    def set_start_method(self, _method: str, force: bool = False) -> None:
        if not force:
            msg = "force must be True"
            raise AssertionError(msg)

    def get_context(self, method: str | None = None) -> _DummyContext:
        if method is None:
            return self.default_context
        if method == "fork":
            if not self.fork_available:
                msg = "fork not available"
                raise ValueError(msg)
            return self.fork_context
        if method == "spawn":
            return self.spawn_context
        msg = f"Unexpected method: {method}"
        raise AssertionError(msg)


def test_plugin_resolves_fork_for_linux_forkserver() -> None:
    mp = _DummyMultiprocessing(default_method="forkserver", fork_available=True)

    ctx = SAQPlugin._resolve_multiprocessing_context(mp, "Linux")

    assert ctx.get_start_method() == "fork"


def test_plugin_falls_back_when_fork_unavailable() -> None:
    mp = _DummyMultiprocessing(default_method="forkserver", fork_available=False)

    ctx = SAQPlugin._resolve_multiprocessing_context(mp, "Linux")

    assert ctx.get_start_method() == "forkserver"


def test_cli_resolves_spawn_for_darwin() -> None:
    mp = _DummyMultiprocessing(default_method="fork", fork_available=True)

    ctx = resolve_cli_context(mp, "Darwin")

    assert ctx.get_start_method() == "spawn"
