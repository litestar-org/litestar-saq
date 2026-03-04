# ruff: noqa: BLE001
# pyright: reportMissingImports=false
# mypy: disable-error-code="import-not-found"
from collections.abc import AsyncGenerator, Collection, MutableMapping
from dataclasses import dataclass, field
from datetime import timezone, tzinfo
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, TypedDict, TypeVar, Union, cast

from litestar.exceptions import ImproperlyConfiguredException
from litestar.serialization import decode_json, encode_json
from litestar.utils.module_loader import import_string, module_to_os_path
from saq.queue.base import Queue
from saq.types import Context, DumpType, LoadType, PartialTimersDict, QueueInfo, ReceivesContext, WorkerInfo
from typing_extensions import NotRequired

from litestar_saq.base import CronJob, Job, JsonDict, Worker
from litestar_saq.typing import OPENTELEMETRY_INSTALLED

if TYPE_CHECKING:
    from litestar import Litestar
    from litestar.types.callable_types import Guard  # pyright: ignore[reportUnknownVariableType]
    from saq.types import Function

T = TypeVar("T")


def serializer(value: Any) -> str:
    """Serialize JSON field values.

    Args:
        value: Any json serializable value.

    Returns:
        JSON string.

    """
    return encode_json(value).decode("utf-8")


def _get_static_files() -> Path:
    return Path(module_to_os_path("saq") / "web" / "static")


@dataclass
class TaskQueues:
    """Task queues."""

    queues: "MutableMapping[str, Queue]" = field(default_factory=dict)  # pyright: ignore

    def get(self, name: str) -> Queue:
        """Get a queue by name.

        Args:
            name: The name of the queue.

        Returns:
            The queue.

        Raises:
            ImproperlyConfiguredException: If the queue does not exist.

        """
        queue = self.queues.get(name)
        if queue is not None:
            return queue
        msg = "Could not find the specified queue. Please check your configuration."
        raise ImproperlyConfiguredException(msg)


@dataclass
class SAQConfig:
    """SAQ Configuration."""

    queue_configs: "Collection[QueueConfig]" = field(default_factory=list)  # pyright: ignore
    """Configuration for Queues"""

    queue_instances: "Optional[MutableMapping[str, Queue]]" = None
    """Current configured queue instances. When None, queues will be auto-created on startup"""
    queues_dependency_key: str = field(default="task_queues")
    """Key to use for storing dependency information in litestar."""
    worker_processes: int = 1
    """The number of worker processes to spawn.

    Default is set to 1.
    """
    app_path: "Optional[str]" = None
    """Litestar app path to use when spawning worker processes.

    This is required on platforms that must use the "spawn" start method
    (e.g. macOS) unless the `LITESTAR_APP` environment variable is set.
    """

    json_deserializer: LoadType = decode_json
    """This is a Python callable that will
    convert a JSON string to a Python object. By default, this is set to Litestar's
    :attr:`decode_json() <.serialization.decode_json>` function."""
    json_serializer: DumpType = serializer
    """This is a Python callable that will render a given object as JSON.
    By default, Litestar's :attr:`encode_json() <.serialization.encode_json>` is used."""
    static_files: Path = field(default_factory=_get_static_files)
    """Location of the static files to serve for the SAQ UI"""
    web_enabled: bool = False
    """If true, the worker admin UI is launched on worker startup.."""
    web_path: str = "/saq"
    """Base path to serve the SAQ web UI"""
    web_guards: "Optional[list[Guard]]" = field(default=None)
    """Guards to apply to web endpoints."""
    web_include_in_schema: bool = False
    """Include Queue API endpoints in generated OpenAPI schema"""
    use_server_lifespan: bool = False
    """Utilize the server lifespan hook to run SAQ."""

    # OpenTelemetry configuration
    enable_otel: "Optional[bool]" = None
    """Enable OpenTelemetry instrumentation.

    - None (default): Auto-detect - enabled if opentelemetry is installed
      AND Litestar OpenTelemetryPlugin is present
    - True: Force enable (raises error if not installed)
    - False: Force disable
    """
    otel_tracer_name: str = "litestar_saq"
    """Name to use when getting the OpenTelemetry tracer."""

    def should_enable_otel(self, app: "Optional[Litestar]" = None) -> bool:
        """Determine if OTEL instrumentation should be enabled.

        Args:
            app: Optional Litestar app to check for OpenTelemetryPlugin.

        Returns:
            True if OTEL instrumentation should be enabled.

        Raises:
            ImproperlyConfiguredException: If enable_otel=True but opentelemetry not installed.
        """
        if self.enable_otel is True:
            if not OPENTELEMETRY_INSTALLED:
                msg = (
                    "enable_otel=True but opentelemetry is not installed. Install with: pip install litestar-saq[otel]"
                )
                raise ImproperlyConfiguredException(msg)
            return True
        if self.enable_otel is False:
            return False

        if not OPENTELEMETRY_INSTALLED:
            return False

        if app is not None:
            return _has_otel_plugin(app)

        return False

    @property
    def signature_namespace(self) -> "dict[str, Any]":
        """Return the plugin's signature namespace.

        Returns:
            A string keyed dict of names to be added to the namespace for signature forward reference resolution.
        """
        return {
            "Queue": Queue,
            "Worker": Worker,
            "QueueInfo": QueueInfo,
            "WorkerInfo": WorkerInfo,
            "Job": Job,
            "TaskQueues": TaskQueues,
        }

    async def provide_queues(self) -> "AsyncGenerator[TaskQueues, None]":
        """Provide the configured job queues.

        Yields:
            The configured job queues.
        """
        queues = self.get_queues()
        for queue in queues.queues.values():
            await queue.connect()
        yield queues

    def filter_delete_queues(self, queues: "list[str]") -> None:
        """Remove all queues except the ones in the given list."""
        new_config = [queue_config for queue_config in self.queue_configs if queue_config.name in queues]
        self.queue_configs = new_config
        if self.queue_instances is not None:
            for queue_name in dict(self.queue_instances):
                if queue_name not in queues:
                    del self.queue_instances[queue_name]

    def get_queues(self) -> TaskQueues:
        """Get the configured SAQ queues.

        Returns:
            The configured job queues.
        """
        if self.queue_instances is not None:
            return TaskQueues(queues=self.queue_instances)

        self.queue_instances = {}
        for c in self.queue_configs:
            self.queue_instances[c.name] = c.queue_class(  # type: ignore
                c.get_broker(),
                name=c.name,  # pyright: ignore[reportCallIssue]
                dump=self.json_serializer,
                load=self.json_deserializer,
                **c._broker_options,  # pyright: ignore[reportArgumentType,reportPrivateUsage]  # noqa: SLF001
            )
            queue = self.queue_instances[c.name]
            if hasattr(queue, "_is_pool_provided"):
                queue._is_pool_provided = False  # pyright: ignore[reportAttributeAccessIssue]  # noqa: SLF001
        return TaskQueues(queues=self.queue_instances)


class RedisQueueOptions(TypedDict, total=False):
    """Options for the Redis backend."""

    max_concurrent_ops: NotRequired[int]
    """Maximum concurrent operations. (default 15)
        This throttles calls to `enqueue`, `job`, and `abort` to prevent the Queue
        from consuming too many Redis connections."""
    swept_error_message: NotRequired[str]


class PostgresQueueOptions(TypedDict, total=False):
    """Options for the Postgres backend."""

    versions_table: NotRequired[str]
    jobs_table: NotRequired[str]
    stats_table: NotRequired[str]
    min_size: NotRequired[int]
    max_size: NotRequired[int]
    saq_lock_keyspace: NotRequired[int]
    job_lock_keyspace: NotRequired[int]
    job_lock_sweep: NotRequired[bool]
    priorities: NotRequired["tuple[int, int]"]
    swept_error_message: NotRequired[str]
    manage_pool_lifecycle: NotRequired[bool]


@dataclass
class QueueConfig:
    """SAQ Queue Configuration"""

    dsn: "Optional[str]" = None
    """DSN for connecting to backend. e.g. 'redis://...' or 'postgres://...'.
    """

    broker_instance: "Optional[Any]" = None
    """An instance of a supported saq backend connection..
    """
    id: "Optional[str]" = None
    """An optional ID to supply for the worker."""
    name: str = "default"
    """The name of the queue to create."""
    concurrency: int = 10
    """Number of jobs to process concurrently."""
    broker_options: "Union[RedisQueueOptions, PostgresQueueOptions, dict[str, Any]]" = field(default_factory=dict)  # pyright: ignore
    """Broker-specific options. For Redis or Postgres backends."""
    broker_instance_options: "dict[str, Any]" = field(default_factory=dict)  # pyright: ignore
    """Broker-specific options. For Redis, Postgres or HTTP backends."""
    tasks: "Collection[Union[Function[Context], tuple[str, Function[Context]], str]]" = field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list
    )
    """Allowed list of functions to execute in this queue."""
    scheduled_tasks: "Collection[CronJob]" = field(default_factory=list)  # pyright: ignore
    """Scheduled cron jobs to execute in this queue."""
    cron_tz: "tzinfo" = timezone.utc
    """Timezone for cron jobs."""
    startup: "Optional[Union[ReceivesContext[Context], str, Collection[Union[ReceivesContext[Context], str]]]]" = None
    """Async callable to call on startup."""
    shutdown: "Optional[Union[ReceivesContext[Context], str, Collection[Union[ReceivesContext[Context], str]]]]" = None
    """Async callable to call on shutdown."""
    before_process: "Optional[Union[ReceivesContext[Context], str, Collection[Union[ReceivesContext[Context], str]]]]" = None
    """Async callable to call before a job processes."""
    after_process: "Optional[Union[ReceivesContext[Context], str, Collection[Union[ReceivesContext[Context], str]]]]" = None
    """Async callable to call after a job processes."""
    timers: "Optional[PartialTimersDict]" = None
    """Dict with various timer overrides in seconds
            schedule: how often we poll to schedule jobs
            stats: how often to update stats
            sweep: how often to clean up stuck jobs
            abort: how often to check if a job is aborted"""
    dequeue_timeout: float = 0
    """How long to wait to dequeue."""
    burst: bool = False
    """If True, the worker will process jobs in burst mode."""
    max_burst_jobs: "Optional[int]" = None
    """The maximum number of jobs to process in burst mode."""
    metadata: "Optional[JsonDict]" = None
    """Arbitrary data to pass to the worker which it will register with saq."""
    multiprocessing_mode: 'Literal["multiprocessing", "threading"]' = "multiprocessing"
    """Executes with the multiprocessing or threading backend. Multi-processing is recommended and how SAQ is designed to work."""
    separate_process: bool = True
    """Executes as a separate event loop when True.
            Set it False to execute within the Litestar application."""
    shutdown_grace_period_s: "Optional[int]" = None
    """Time in seconds to allow jobs to complete gracefully before forced shutdown.

    When the worker receives a shutdown signal, it will wait up to this duration
    for running jobs to complete naturally before forcing cancellation.
    If None, uses SAQ's default.
    """
    cancellation_hard_deadline_s: "Optional[float]" = None
    """Absolute deadline in seconds for task cancellation.

    After this time, tasks are forcibly terminated regardless of grace period.
    This prevents zombie tasks from blocking shutdown indefinitely.
    If None, uses SAQ's default (1.0s).
    """
    poll_interval: "Optional[float]" = None
    """Queue polling interval in seconds.

    Controls how frequently the worker checks the queue for new jobs.
    Lower values = lower latency but higher CPU/database load.
    Higher values = higher latency but lower resource consumption.
    If None, uses SAQ's backend-specific default.
    """

    def __post_init__(self) -> None:
        if getattr(self, "_normalized", False):
            return

        if self.dsn and self.broker_instance:
            msg = "Cannot specify both `dsn` and `broker_instance`"
            raise ImproperlyConfiguredException(msg)
        if not self.dsn and not self.broker_instance:
            msg = "Must specify either `dsn` or `broker_instance`"
            raise ImproperlyConfiguredException(msg)
        self.tasks = [self._get_or_import_task(task) for task in self.tasks]
        if self.startup is not None and not isinstance(self.startup, Collection):
            self.startup = [self.startup]
        if self.shutdown is not None and not isinstance(self.shutdown, Collection):
            self.shutdown = [self.shutdown]
        if self.before_process is not None and not isinstance(self.before_process, Collection):
            self.before_process = [self.before_process]
        if self.after_process is not None and not isinstance(self.after_process, Collection):
            self.after_process = [self.after_process]
        self.startup = [self._get_or_import_task(task) for task in self.startup or []]  # pyright: ignore
        self.shutdown = [self._get_or_import_task(task) for task in self.shutdown or []]  # pyright: ignore
        self.before_process = [self._get_or_import_task(task) for task in self.before_process or []]  # pyright: ignore
        self.after_process = [self._get_or_import_task(task) for task in self.after_process or []]  # pyright: ignore
        self._broker_type: Optional[Literal["redis", "postgres", "http"]] = None
        self._queue_class: Optional[type[Queue]] = None
        self._normalized = True

    def get_broker(self) -> "Any":
        """Get the configured Broker connection.

        Raises:
            ImproperlyConfiguredException: If the broker type is invalid.

        Returns:
            Dictionary of queues.
        """

        if self.broker_instance is not None:
            self._ensure_postgres_pool_defaults()
            return self.broker_instance

        if self.dsn and self.dsn.startswith("redis"):
            from redis.asyncio import from_url as redis_from_url  # pyright: ignore[reportUnknownVariableType]
            from saq.queue.redis import RedisQueue

            self.broker_instance = redis_from_url(self.dsn, **self.broker_instance_options)
            self._broker_type = "redis"
            self._queue_class = RedisQueue
        elif self.dsn and self.dsn.startswith("postgresql"):
            from psycopg_pool import AsyncConnectionPool
            from saq.queue.postgres import PostgresQueue

            self.broker_instance = AsyncConnectionPool(
                self.dsn,
                check=AsyncConnectionPool.check_connection,  # pyright: ignore[reportUnknownMemberType]
                open=False,
                **self.broker_instance_options,
            )
            self._ensure_postgres_pool_defaults()
            self._broker_type = "postgres"
            self._queue_class = PostgresQueue
        elif self.dsn and self.dsn.startswith("http"):
            from saq.queue.http import HttpQueue

            self.broker_instance = HttpQueue(self.dsn, **self.broker_instance_options)
            self._broker_type = "http"
            self._queue_class = HttpQueue
        else:
            msg = "Invalid broker type"
            raise ImproperlyConfiguredException(msg)
        return self.broker_instance

    def _ensure_postgres_pool_defaults(self) -> None:
        """Normalize psycopg connection pool defaults for SAQ.

        SAQ's PostgresQueue expects ``pool.kwargs`` to be a mutable mapping with
        ``autocommit`` enabled. psycopg 3.2+ may default ``pool.kwargs`` to
        ``None``, which raises an ``AttributeError`` in SAQ >=0.24.4 when it
        checks ``pool.kwargs.get``. Setting a dict here keeps compatibility with
        older SAQ releases while satisfying newer versions' autocommit guard.
        """

        if self.broker_instance is None:
            return

        if not self._is_instance_of(self.broker_instance, "psycopg_pool", "AsyncConnectionPool"):
            return

        kwargs: Optional[MutableMapping[str, Any]] = getattr(self.broker_instance, "kwargs", None)
        if kwargs is None:
            self.broker_instance.kwargs = {}
            kwargs = self.broker_instance.kwargs  # pyright: ignore

        kwargs.setdefault("autocommit", True)  # pyright: ignore

    @staticmethod
    def _is_instance_of(obj: object, module_path: str, class_name: str) -> bool:
        try:
            module = import_module(module_path)
            cls = getattr(module, class_name)
        except Exception:
            return False
        return isinstance(obj, cls)

    @property
    def broker_type(self) -> 'Literal["redis", "postgres", "http"]':
        """Type of broker to use.

        Raises:
            ImproperlyConfiguredException: If the broker type is invalid.

        Returns:
            The broker type.
        """
        if self._broker_type is None and self.broker_instance is not None:
            if self._is_instance_of(self.broker_instance, "psycopg_pool", "AsyncConnectionPool"):
                self._broker_type = "postgres"
            elif self._is_instance_of(self.broker_instance, "redis.asyncio", "Redis") or self._is_instance_of(
                self.broker_instance, "redis", "Redis"
            ):
                self._broker_type = "redis"
            elif self._is_instance_of(self.broker_instance, "saq.queue.http", "HttpQueue"):
                self._broker_type = "http"
        if self._broker_type is None:
            self.get_broker()
        if self._broker_type is None:
            msg = "Invalid broker type"
            raise ImproperlyConfiguredException(msg)
        return self._broker_type

    @property
    def _broker_options(self) -> "Union[RedisQueueOptions, PostgresQueueOptions, dict[str, Any]]":
        """Broker-specific options.

        Returns:
            The broker options.
        """
        if self._broker_type == "postgres" and "manage_pool_lifecycle" not in self.broker_options:
            self.broker_options["manage_pool_lifecycle"] = True  # type: ignore[typeddict-unknown-key]
        return self.broker_options

    @property
    def queue_class(self) -> "type[Queue]":
        """Type of queue to use.

        Raises:
            ImproperlyConfiguredException: If the queue class is invalid.

        Returns:
            The queue class.
        """
        if self._queue_class is None and self.broker_instance is not None:
            if self._is_instance_of(self.broker_instance, "psycopg_pool", "AsyncConnectionPool"):
                from saq.queue.postgres import PostgresQueue

                self._queue_class = PostgresQueue
            elif self._is_instance_of(self.broker_instance, "redis.asyncio", "Redis") or self._is_instance_of(
                self.broker_instance, "redis", "Redis"
            ):
                from saq.queue.redis import RedisQueue

                self._queue_class = RedisQueue
            elif self._is_instance_of(self.broker_instance, "saq.queue.http", "HttpQueue"):
                from saq.queue.http import HttpQueue

                self._queue_class = HttpQueue
        if self._queue_class is None:
            self.get_broker()
        if self._queue_class is None:
            msg = "Invalid queue class"
            raise ImproperlyConfiguredException(msg)
        return self._queue_class

    @staticmethod
    def _get_or_import_task(
        task_or_import_string: "Union[str, tuple[str, Function[Context]], ReceivesContext[Context]]",
    ) -> "ReceivesContext[Context]":
        """Get or import a task.

        Args:
            task_or_import_string: The task or import string.

        Returns:
            The task.
        """
        if isinstance(task_or_import_string, str):
            return cast("ReceivesContext[Context]", import_string(task_or_import_string))
        if isinstance(task_or_import_string, tuple):
            return task_or_import_string[1]  # pyright: ignore
        return task_or_import_string


def _has_otel_plugin(app: "Litestar") -> bool:
    """Check if Litestar app has OpenTelemetryPlugin configured.

    Args:
        app: Litestar application instance.

    Returns:
        True if OpenTelemetryPlugin is present in app plugins.
    """
    try:
        from litestar.plugins.opentelemetry import (
            OpenTelemetryPlugin,  # pyright: ignore[reportAttributeAccessIssue,reportMissingImports,reportUnknownVariableType]
        )

        return app.plugins.get(OpenTelemetryPlugin) is not None  # pyright: ignore[reportUnknownArgumentType]
    except ImportError:
        return False
