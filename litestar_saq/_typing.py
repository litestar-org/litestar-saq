# ruff: noqa: RUF100, PLR0913, A002, ARG001, ARG002, B903, PLR0917
# mypy: disable-error-code="misc,assignment"
"""Wrapper around library classes for compatibility when libraries are not installed.

This module provides stub implementations of optional dependency types that allow
litestar-saq to use these types in type hints and runtime code without requiring
the packages to be installed.

Pattern inspired by sqlspec/_typing.py

IMPORTANT: This is a PRIVATE module. All application code MUST import from
`litestar_saq.typing`, NEVER from `litestar_saq._typing`.
"""

from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any, Optional

from typing_extensions import Self

__all__ = (
    "OPENTELEMETRY_INSTALLED",
    "STRUCTLOG_INSTALLED",
    "Context",
    "Span",
    "SpanKind",
    "Status",
    "StatusCode",
    "Tracer",
    "propagate",
    "trace",
)

STRUCTLOG_INSTALLED: bool = find_spec("structlog") is not None


class SpanStub:
    """Placeholder implementation for opentelemetry.trace.Span."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute (no-op when OTEL not installed)."""
        return

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        """Set multiple span attributes (no-op when OTEL not installed)."""
        return

    def add_event(
        self,
        name: str,
        attributes: Optional[Mapping[str, Any]] = None,
        timestamp: Optional[int] = None,
    ) -> None:
        """Add an event to the span (no-op when OTEL not installed)."""
        return

    def record_exception(
        self,
        exception: BaseException,
        attributes: Optional[Mapping[str, Any]] = None,
        timestamp: Optional[int] = None,
        escaped: bool = False,
    ) -> None:
        """Record an exception (no-op when OTEL not installed)."""
        return

    def set_status(
        self,
        status: Any,
        description: Optional[str] = None,
    ) -> None:
        """Set span status (no-op when OTEL not installed)."""
        return

    def end(self, end_time: Optional[int] = None) -> None:
        """End the span (no-op when OTEL not installed)."""
        return

    def get_span_context(self) -> Any:
        """Get span context (returns None when OTEL not installed).

        Returns:
            None when OTEL not installed, SpanContext otherwise.
        """
        return None

    def is_recording(self) -> bool:
        """Check if span is recording (returns False when OTEL not installed).

        Returns:
            False when OTEL not installed.
        """
        return False

    def __enter__(self) -> Self:
        """Context manager entry.

        Returns:
            Self for use in context manager.
        """
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        return


class TracerStub:
    """Placeholder implementation for opentelemetry.trace.Tracer."""

    def start_span(
        self,
        name: str,
        context: Any = None,
        kind: Any = None,
        attributes: Any = None,
        links: Any = None,
        start_time: Any = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> SpanStub:
        """Start a new span (returns stub when OTEL not installed).

        Returns:
            SpanStub instance.
        """
        return SpanStub()

    def start_as_current_span(
        self,
        name: str,
        context: Any = None,
        kind: Any = None,
        attributes: Any = None,
        links: Any = None,
        start_time: Any = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> SpanStub:
        """Start span as current (returns stub when OTEL not installed).

        Returns:
            SpanStub instance.
        """
        return SpanStub()


class SpanKindStub:
    """Placeholder for opentelemetry.trace.SpanKind enum."""

    INTERNAL = 0
    SERVER = 1
    CLIENT = 2
    PRODUCER = 3
    CONSUMER = 4


class StatusCodeStub:
    """Placeholder for opentelemetry.trace.StatusCode enum."""

    UNSET = 0
    OK = 1
    ERROR = 2


class StatusStub:
    """Placeholder for opentelemetry.trace.Status."""

    def __init__(
        self,
        status_code: Any = None,
        description: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        self.description = description


class _TraceModuleStub:
    """Placeholder for opentelemetry.trace module."""

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        tracer_provider: Any = None,
    ) -> TracerStub:
        """Get a tracer instance (returns stub when OTEL not installed).

        Returns:
            TracerStub instance.
        """
        return TracerStub()

    def get_current_span(self, context: Any = None) -> SpanStub:
        """Get current span (returns stub when OTEL not installed).

        Returns:
            SpanStub instance.
        """
        return SpanStub()

    def set_span_in_context(
        self,
        span: Any,
        context: Any = None,
    ) -> Any:
        """Set span in context (no-op when OTEL not installed).

        Returns:
            The context parameter unchanged.
        """
        return context


class _PropagateModuleStub:
    """Placeholder for opentelemetry.propagate module."""

    def inject(
        self,
        carrier: dict[str, str],
        context: Any = None,
        setter: Any = None,
    ) -> None:
        """Inject trace context into carrier (no-op when OTEL not installed)."""
        return

    def extract(
        self,
        carrier: dict[str, str],
        context: Any = None,
        getter: Any = None,
    ) -> Any:
        """Extract trace context from carrier (returns None when OTEL not installed).

        Returns:
            None when OTEL not installed.
        """
        return None


class _ContextStub:
    """Placeholder for opentelemetry.context.Context."""


try:
    from opentelemetry import propagate as _real_propagate  # pyright: ignore
    from opentelemetry import trace as _real_trace  # pyright: ignore
    from opentelemetry.context import Context as _RealContext  # pyright: ignore
    from opentelemetry.trace import (  # pyright: ignore
        Span as _RealSpan,
    )
    from opentelemetry.trace import (
        SpanKind as _RealSpanKind,
    )
    from opentelemetry.trace import (
        Status as _RealStatus,
    )
    from opentelemetry.trace import (
        StatusCode as _RealStatusCode,
    )
    from opentelemetry.trace import (
        Tracer as _RealTracer,
    )

    Span = _RealSpan  # pyright: ignore
    SpanKind = _RealSpanKind  # pyright: ignore
    Status = _RealStatus  # pyright: ignore
    StatusCode = _RealStatusCode  # pyright: ignore
    Tracer = _RealTracer  # pyright: ignore
    Context = _RealContext  # pyright: ignore
    trace = _real_trace  # pyright: ignore
    propagate = _real_propagate  # pyright: ignore
    OPENTELEMETRY_INSTALLED = True  # pyright: ignore[reportConstantRedefinition]
except ImportError:
    Span = SpanStub
    SpanKind = SpanKindStub
    Status = StatusStub
    StatusCode = StatusCodeStub
    Tracer = TracerStub
    Context = _ContextStub
    trace = _TraceModuleStub()
    propagate = _PropagateModuleStub()
    OPENTELEMETRY_INSTALLED = False  # pyright: ignore[reportConstantRedefinition]
