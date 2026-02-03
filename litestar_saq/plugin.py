# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import signal
import sys
import time
from contextlib import contextmanager
from multiprocessing import Process
from typing import TYPE_CHECKING, Any, Optional, cast

from litestar.plugins import CLIPlugin, InitPluginProtocol

from litestar_saq.base import Worker

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator

    from click import Group
    from litestar import Litestar
    from litestar.config.app import AppConfig
    from saq.queue.base import Queue
    from saq.types import Context, Function, ReceivesContext

    from litestar_saq.config import SAQConfig, TaskQueues


class SAQPlugin(InitPluginProtocol, CLIPlugin):
    """SAQ plugin."""

    __slots__ = ("_config", "_enable_otel", "_otel_tracer", "_processes", "_shutdown_timeout", "_worker_instances")

    DEFAULT_SHUTDOWN_TIMEOUT = 5.0  # seconds when no grace period configured
    SHUTDOWN_BUFFER = 2.0  # extra time for signal propagation
    WORKER_JOIN_TIMEOUT = 1.0  # seconds

    def __init__(self, config: "SAQConfig") -> None:
        """Initialize ``SAQPlugin``.

        Args:
            config: configure and start SAQ.
        """
        self._config = config
        self._worker_instances: Optional[dict[str, Worker]] = None
        self._enable_otel: Optional[bool] = None
        self._otel_tracer: Optional[Any] = None

    @property
    def config(self) -> "SAQConfig":
        return self._config

    def on_cli_init(self, cli: "Group") -> None:
        from litestar_saq.cli import build_cli_app

        cli.add_command(build_cli_app())
        return super().on_cli_init(cli)  # type: ignore[safe-super]

    def on_app_init(self, app_config: "AppConfig") -> "AppConfig":
        """Configure application for use with SQLAlchemy.

        Args:
            app_config: The :class:`AppConfig <.config.app.AppConfig>` instance.

        Returns:
            The :class:`AppConfig <.config.app.AppConfig>` instance.
        """

        from litestar.di import Provide
        from litestar.static_files import create_static_files_router  # pyright: ignore[reportUnknownVariableType]

        from litestar_saq.controllers import build_controller

        app_config.dependencies.update(
            {self._config.queues_dependency_key: Provide(dependency=self._config.provide_queues)}
        )
        if self._config.web_enabled:
            app_config.route_handlers.append(
                create_static_files_router(
                    directories=[self._config.static_files],
                    path=f"{self._config.web_path}/static",
                    name="saq",
                    html_mode=False,
                    opt={"exclude_from_auth": True},
                    include_in_schema=False,
                ),
            )
            app_config.route_handlers.append(
                build_controller(self._config.web_path, self._config.web_guards, self._config.web_include_in_schema),
            )
        app_config.signature_namespace.update(self._config.signature_namespace)

        workers = self.get_workers()
        for worker in workers.values():
            app_config.on_startup.append(worker.on_app_startup)
            app_config.on_shutdown.append(worker.on_app_shutdown)
        app_config.on_shutdown.extend([self.remove_workers])
        return app_config

    def get_workers(self) -> "dict[str, Worker]":
        """Return workers"""
        if self._worker_instances is not None:
            return self._worker_instances

        if self._enable_otel is None:
            self._enable_otel = self._config.should_enable_otel()

        otel_tracer = self._get_otel_tracer() if self._enable_otel else None

        self._worker_instances = {
            queue_config.name: Worker(
                queue=self.get_queue(queue_config.name),
                id=queue_config.id,
                functions=cast("Collection[Function[Context]]", queue_config.tasks),
                cron_jobs=queue_config.scheduled_tasks,
                cron_tz=queue_config.cron_tz,
                concurrency=queue_config.concurrency,
                startup=cast("Collection[ReceivesContext[Context]]", queue_config.startup),
                shutdown=cast("Collection[ReceivesContext[Context]]", queue_config.shutdown),
                before_process=cast("Collection[ReceivesContext[Context]]", queue_config.before_process),
                after_process=cast("Collection[ReceivesContext[Context]]", queue_config.after_process),
                timers=queue_config.timers,
                dequeue_timeout=queue_config.dequeue_timeout,
                separate_process=queue_config.separate_process,
                burst=queue_config.burst,
                max_burst_jobs=queue_config.max_burst_jobs,
                metadata=queue_config.metadata,
                shutdown_grace_period_s=queue_config.shutdown_grace_period_s,
                cancellation_hard_deadline_s=queue_config.cancellation_hard_deadline_s,
                poll_interval=queue_config.poll_interval,
                enable_otel=self._enable_otel,
                otel_tracer=otel_tracer,
            )
            for queue_config in self._config.queue_configs
        }

        return self._worker_instances

    def _get_otel_tracer(self) -> Any:
        """Get or create the OTEL tracer."""
        if self._otel_tracer is None:
            from litestar_saq.instrumentation import get_tracer

            self._otel_tracer = get_tracer(self._config.otel_tracer_name)
        return self._otel_tracer

    def remove_workers(self) -> None:
        self._worker_instances = None

    def _get_shutdown_timeout(self) -> float:
        """Calculate the shutdown timeout from worker configurations.

        Returns the maximum shutdown_grace_period_s from all workers plus
        buffer time. Falls back to DEFAULT_SHUTDOWN_TIMEOUT if no grace
        periods are configured.

        Returns:
            Shutdown timeout in seconds.
        """
        workers = self.get_workers().values()
        grace_periods: list[float] = []
        for worker in workers:
            grace_period = getattr(worker, "_shutdown_grace_period_s", None)
            if grace_period is not None:
                grace_periods.append(grace_period)
        if grace_periods:
            return max(grace_periods) + self.SHUTDOWN_BUFFER
        return self.DEFAULT_SHUTDOWN_TIMEOUT

    def get_queues(self) -> "TaskQueues":
        return self._config.get_queues()

    def get_queue(self, name: str) -> "Queue":
        return self.get_queues().get(name)

    @contextmanager
    def server_lifespan(self, app: "Litestar") -> "Iterator[None]":
        import multiprocessing
        import platform

        from litestar.cli._utils import console  # pyright: ignore

        from litestar_saq.cli import run_saq_worker

        if platform.system() == "Darwin":
            multiprocessing.set_start_method("fork", force=True)

        if not self._config.use_server_lifespan:
            yield
            return

        console.rule("[yellow]Starting SAQ Workers[/]", align="left")
        self._processes: list[Process] = []
        self._shutdown_timeout = self._get_shutdown_timeout()

        def handle_shutdown(_signum: Any, _frame: Any) -> None:
            """Handle shutdown signals gracefully."""
            console.print(
                f"[yellow]Received shutdown signal, stopping workers (timeout: {self._shutdown_timeout:.1f}s)...[/]"
            )
            self._terminate_workers(self._processes, timeout=self._shutdown_timeout)
            sys.exit(0)

        # Register signal handlers
        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)

        try:
            for worker_name, worker in self.get_workers().items():
                for i in range(self.config.worker_processes):
                    console.print(f"[yellow]Starting worker process {i + 1} for {worker_name}[/]")
                    process = Process(
                        target=run_saq_worker,
                        args=(
                            worker,
                            app.logging_config,
                        ),
                        name=f"worker-{worker_name}-{i + 1}",
                    )
                    process.start()
                    self._processes.append(process)

            yield

        except Exception as e:
            console.print(f"[red]Error in worker processes: {e}[/]")
            raise
        finally:
            console.print(f"[yellow]Shutting down SAQ workers (timeout: {self._shutdown_timeout:.1f}s)...[/]")
            self._terminate_workers(self._processes, timeout=self._shutdown_timeout)
            console.print("[yellow]SAQ workers stopped.[/]")

    @staticmethod
    def _terminate_workers(processes: "list[Process]", timeout: float = DEFAULT_SHUTDOWN_TIMEOUT) -> None:
        """Gracefully terminate worker processes with timeout.

        Args:
            processes: List of worker processes to terminate
            timeout: Maximum time to wait for graceful shutdown in seconds.
                Should be at least as long as the worker's shutdown_grace_period_s
                plus buffer time for signal propagation.
        """
        # Send SIGTERM to all processes
        from litestar.cli._utils import console  # pyright: ignore

        for p in processes:
            if p.is_alive():
                p.terminate()

        # Wait for processes to terminate gracefully
        termination_start = time.time()
        while time.time() - termination_start < timeout:
            if not any(p.is_alive() for p in processes):
                break
            time.sleep(0.1)

        # Force kill any remaining processes
        for p in processes:
            if p.is_alive():
                try:
                    p.kill()  # Send SIGKILL
                    p.join(timeout=1.0)
                except Exception as e:  # noqa: BLE001
                    console.print(f"[red]Error killing worker process: {e}[/]")
