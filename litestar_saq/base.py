# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportInvalidTypeForm=false, reportUnknownArgumentType=false
# mypy: disable-error-code="unused-ignore,assignment"
import asyncio
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import timezone, tzinfo
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

from litestar.utils.module_loader import import_string
from saq import Job as SaqJob
from saq.job import CronJob as SaqCronJob
from saq.types import Context
from saq.worker import Worker as SaqWorker

from litestar_saq.heartbeat import HeartbeatManager
from litestar_saq.typing import OPENTELEMETRY_INSTALLED, Span, Tracer

if TYPE_CHECKING:
    from saq.queue.base import Queue
    from saq.types import Function, PartialTimersDict, ReceivesContext

JsonDict = dict[str, Any]
_OTEL_SPAN_KEY = "_otel_span"


def _normalize_hooks(
    hooks: "Optional[Union[ReceivesContext[Context], Collection[ReceivesContext[Context]]]]",
) -> "list[ReceivesContext[Context]]":
    """Normalize hooks to a list.

    Args:
        hooks: Single hook, collection of hooks, or None.

    Returns:
        List of hooks (empty list if None).
    """
    if hooks is None:
        return []
    if isinstance(hooks, Collection) and not callable(hooks):
        return list(hooks)
    return [hooks]  # type: ignore[list-item]


@dataclass
class Job(SaqJob):
    """Job Details"""


@dataclass
class CronJob(SaqCronJob[Context]):
    """Cron Job Details"""

    function: "Union[Function[Context], str]"  # type: ignore[assignment]
    meta: "dict[str, Any]" = field(default_factory=dict)  # pyright: ignore

    def __post_init__(self) -> None:
        self.function = self._get_or_import_function(self.function)  # pyright: ignore[reportIncompatibleMethodOverride]

    @staticmethod
    def _get_or_import_function(
        function_or_import_string: "Union[str, Function[Context]]",
    ) -> "Function[Context]":
        if isinstance(function_or_import_string, str):
            return cast("Function[Context]", import_string(function_or_import_string))
        return function_or_import_string


class Worker(SaqWorker[Context]):
    """Worker.

    Extends SAQ's Worker with Litestar lifecycle integration.

    Args:
        queue: SAQ Queue instance.
        functions: Task functions to register.
        id: Optional worker identifier.
        concurrency: Number of concurrent tasks (default: 10).
        cron_jobs: Scheduled cron jobs.
        cron_tz: Timezone for cron jobs (default: UTC).
        startup: Async callable(s) to run on worker startup.
        shutdown: Async callable(s) to run on worker shutdown.
        before_process: Async callable(s) to run before each job.
        after_process: Async callable(s) to run after each job.
        timers: Dict with timer overrides (schedule, stats, sweep, abort).
        dequeue_timeout: How long to wait for dequeue (default: 0).
        burst: If True, process jobs in burst mode.
        max_burst_jobs: Maximum jobs in burst mode.
        metadata: Arbitrary metadata to register with SAQ.
        separate_process: Execute as separate event loop (default: True).
        multiprocessing_mode: Backend for multiprocessing (default: "multiprocessing").
        shutdown_grace_period_s: Time in seconds to allow jobs to complete
            gracefully before forced shutdown. Defaults to SAQ's internal default.
        cancellation_hard_deadline_s: Absolute deadline for task cancellation.
            Prevents zombie tasks from blocking shutdown. Defaults to SAQ's default (1.0s).
        poll_interval: Queue polling interval in seconds. Lower values reduce
            latency but increase CPU/database load. Defaults to backend-specific value.
        enable_otel: Enable OpenTelemetry instrumentation for job processing.
        otel_tracer: Optional custom tracer instance. If not provided and enable_otel
            is True, a default tracer will be created.
        enable_heartbeat_manager: Enable centralized HeartbeatManager for batched
            heartbeat updates (default: True). The manager is injected into job context
            and used by the @monitored_job decorator for efficient batched updates.
        heartbeat_flush_interval: Seconds between heartbeat batch flushes when
            HeartbeatManager is enabled. Default: 30.0.
    """

    def __init__(
        self,
        queue: "Queue",
        functions: "Collection[Union[Function[Context], tuple[str, Function[Context]]]]",
        *,
        id: "Optional[str]" = None,  # noqa: A002
        concurrency: int = 10,
        cron_jobs: "Optional[Collection[CronJob]]" = None,
        cron_tz: "tzinfo" = timezone.utc,
        startup: "Optional[Union[ReceivesContext[Context], Collection[ReceivesContext[Context]]]]" = None,
        shutdown: "Optional[Union[ReceivesContext[Context], Collection[ReceivesContext[Context]]]]" = None,
        before_process: "Optional[Union[ReceivesContext[Context], Collection[ReceivesContext[Context]]]]" = None,
        after_process: "Optional[Union[ReceivesContext[Context], Collection[ReceivesContext[Context]]]]" = None,
        timers: "Optional[PartialTimersDict]" = None,
        dequeue_timeout: float = 0,
        burst: bool = False,
        max_burst_jobs: "Optional[int]" = None,
        metadata: "Optional[JsonDict]" = None,
        separate_process: bool = True,
        multiprocessing_mode: Literal["multiprocessing", "threading"] = "multiprocessing",
        shutdown_grace_period_s: "Optional[int]" = None,
        cancellation_hard_deadline_s: "Optional[float]" = None,
        poll_interval: "Optional[float]" = None,
        enable_otel: bool = False,
        otel_tracer: "Optional[Tracer]" = None,
        enable_heartbeat_manager: bool = True,
        heartbeat_flush_interval: float = 30.0,
    ) -> None:
        self.separate_process = separate_process
        self.multiprocessing_mode = multiprocessing_mode
        self._enable_otel = enable_otel and OPENTELEMETRY_INSTALLED
        self._otel_tracer = otel_tracer
        self._enable_heartbeat_manager = enable_heartbeat_manager
        self._heartbeat_manager: Optional[HeartbeatManager] = None
        self._heartbeat_flush_interval = heartbeat_flush_interval

        otel_before: list[Any] = [self._otel_before_process] if self._enable_otel else []
        otel_after: list[Any] = [self._otel_after_process] if self._enable_otel else []
        heartbeat_before: list[Any] = [self._inject_heartbeat_manager] if enable_heartbeat_manager else []
        user_before = _normalize_hooks(before_process)
        user_after = _normalize_hooks(after_process)

        kwargs: dict[str, Any] = {
            "id": id,
            "concurrency": concurrency,
            "cron_jobs": cron_jobs,
            "cron_tz": cron_tz,
            "startup": startup,
            "shutdown": shutdown,
            "before_process": otel_before + heartbeat_before + user_before,
            "after_process": user_after + otel_after,
            "timers": timers,
            "dequeue_timeout": dequeue_timeout,
            "burst": burst,
            "max_burst_jobs": max_burst_jobs,
            "metadata": metadata,
        }

        if shutdown_grace_period_s is not None:
            kwargs["shutdown_grace_period_s"] = shutdown_grace_period_s
        if cancellation_hard_deadline_s is not None:
            kwargs["cancellation_hard_deadline_s"] = cancellation_hard_deadline_s
        if poll_interval is not None:
            kwargs["poll_interval"] = poll_interval

        super().__init__(queue, functions, **kwargs)

    def _get_otel_tracer(self) -> Tracer:
        """Get the OTEL tracer, creating one if needed.

        Returns:
            Tracer instance.
        """
        if self._otel_tracer is None:
            from litestar_saq.instrumentation import get_tracer

            self._otel_tracer = get_tracer()
        return self._otel_tracer

    async def _otel_before_process(self, ctx: Context) -> None:
        """OTEL hook called before job processing - creates span."""
        from litestar_saq.instrumentation import create_process_span

        tracer = self._get_otel_tracer()
        span = create_process_span(ctx, tracer, self.queue.name)
        ctx[_OTEL_SPAN_KEY] = span  # type: ignore[literal-required]

    async def _otel_after_process(self, ctx: Context) -> None:
        """OTEL hook called after job processing - ends span."""
        from litestar_saq.instrumentation import end_process_span

        span: Optional[Span] = ctx.get(_OTEL_SPAN_KEY)  # type: ignore[arg-type]
        error: Optional[BaseException] = ctx.get("exception")
        end_process_span(ctx, span, error)

    async def _inject_heartbeat_manager(self, ctx: Context) -> None:
        """Inject HeartbeatManager into job context for monitored_job decorator."""
        if self._heartbeat_manager is not None:
            ctx["heartbeat_manager"] = self._heartbeat_manager  # type: ignore[literal-required,typeddict-unknown-key]

    def get_structlog_context(self) -> dict[str, Any]:
        """Build context dictionary for structlog binding.

        Returns:
            Dictionary of context key-value pairs including worker_id,
            queue_name, concurrency, separate_process, and any custom metadata.
        """
        context: dict[str, Any] = {
            "worker_id": self.id or "unknown",
            "queue_name": self.queue.name,
            "concurrency": self.concurrency,
            "separate_process": self.separate_process,
        }

        # Add custom metadata with prefix to avoid collisions
        if self._metadata:
            for key, value in self._metadata.items():
                context[f"worker_meta_{key}"] = value

        return context

    def configure_structlog_context(self) -> None:
        """Configure structlog context with worker metadata if structlog is installed.

        This method:
        1. Checks if structlog is available
        2. Binds worker-specific context to contextvars
        3. Fails silently if errors occur (context binding is optional)

        Context includes:
        - worker_id: Unique identifier for this worker
        - queue_name: Name of the queue this worker processes
        - concurrency: Number of concurrent jobs
        - separate_process: Whether worker runs in separate process
        - worker_meta_*: Any custom metadata provided in Worker.metadata
        """
        from litestar_saq.typing import STRUCTLOG_INSTALLED

        if not STRUCTLOG_INSTALLED:
            return

        try:
            import structlog  # pyright: ignore[reportMissingImports]

            context = self.get_structlog_context()
            structlog.contextvars.bind_contextvars(**context)  # pyright: ignore[reportUnknownMemberType]
        except Exception as e:  # noqa: BLE001
            # Log at debug level only - context binding is nice-to-have
            import logging

            logging.getLogger(__name__).debug("Failed to configure structlog context: %s", e)

    async def on_app_startup(self) -> None:
        """Attach the worker to the running event loop."""
        # Configure structlog context before starting
        self.configure_structlog_context()

        # Start heartbeat manager if enabled
        if self._enable_heartbeat_manager:
            self._heartbeat_manager = HeartbeatManager(
                queue=self.queue,
                flush_interval=self._heartbeat_flush_interval,
            )
            self._heartbeat_manager.start()

        if not self.separate_process:
            self.SIGNALS = []
            loop = asyncio.get_running_loop()
            self._saq_asyncio_tasks = loop.create_task(self.start())

    async def on_app_shutdown(self) -> None:
        """Stop the worker and cleanup resources."""
        # Stop heartbeat manager if running
        if self._heartbeat_manager is not None:
            self._heartbeat_manager.stop()
            self._heartbeat_manager = None

        if not self.separate_process:
            loop = asyncio.get_running_loop()
            self._saq_asyncio_tasks = loop.create_task(self.stop())
