import contextlib
import os
import signal
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

if TYPE_CHECKING:
    import asyncio
    import multiprocessing
    from collections.abc import Collection

    from click import Group
    from litestar import Litestar
    from litestar.logging.config import BaseLoggingConfig

    from litestar_saq.base import Worker
    from litestar_saq.plugin import SAQPlugin

# Default timeout for graceful shutdown when no grace period is configured
DEFAULT_SHUTDOWN_TIMEOUT = 5.0
# Extra buffer time to allow for signal propagation and cleanup
SHUTDOWN_BUFFER = 2.0


def get_max_shutdown_timeout(workers: "Collection[Worker]") -> float:
    """Calculate the maximum shutdown timeout from worker configurations.

    The timeout is the maximum of all workers' shutdown_grace_period_s + cancellation_hard_deadline_s plus
    a buffer for signal propagation. Falls back to DEFAULT_SHUTDOWN_TIMEOUT
    if no grace periods are configured.

    Args:
        workers: Collection of worker instances.

    Returns:
        Maximum shutdown timeout in seconds.
    """
    grace_periods: list[float] = []
    for worker in workers:
        shutdown_grace_period = getattr(worker, "_shutdown_grace_period_s", 0)
        cancellation_hard_deadline = getattr(worker, "_cancellation_hard_deadline_s", 0)
        grace_period = shutdown_grace_period + cancellation_hard_deadline
        grace_periods.append(grace_period)
    if grace_periods:
        # This ensures that the overall shutdown timeout is always at least 5s
        return max(max(grace_periods) + SHUTDOWN_BUFFER, DEFAULT_SHUTDOWN_TIMEOUT)
    return DEFAULT_SHUTDOWN_TIMEOUT


def _terminate_worker_processes(
    processes: "list[multiprocessing.Process]",
    timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
) -> None:
    """Gracefully terminate worker processes with timeout.

    Args:
        processes: List of worker processes to terminate
        timeout: Maximum time to wait for graceful shutdown in seconds.
            Should be at least as long as the worker's shutdown_grace_period_s
            plus buffer time for signal propagation.
    """
    from litestar.cli._utils import console  # pyright: ignore

    # Send SIGTERM to all processes
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
            except Exception:  # noqa: BLE001
                console.print(f"[red]Error killing worker process: {p.name}[/]")


def _get_event_loop() -> "asyncio.AbstractEventLoop":
    import asyncio

    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def _register_shutdown_handlers(
    loop: "asyncio.AbstractEventLoop",
    shutdown_event: "asyncio.Event",
) -> Callable[[], None]:
    signal_handlers_registered = False
    fallback_handlers_registered = False
    original_sigterm: Any = None
    original_sigint: Any = None

    def request_shutdown() -> None:
        shutdown_event.set()

    try:
        loop.add_signal_handler(signal.SIGTERM, request_shutdown)
        loop.add_signal_handler(signal.SIGINT, request_shutdown)
        signal_handlers_registered = True
    except (NotImplementedError, RuntimeError):
        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        def fallback_handler(_signum: int, _frame: Any) -> None:
            loop.call_soon_threadsafe(shutdown_event.set)

        try:
            signal.signal(signal.SIGTERM, fallback_handler)
            signal.signal(signal.SIGINT, fallback_handler)
            fallback_handlers_registered = True
        except ValueError:
            fallback_handlers_registered = False

    def cleanup() -> None:
        if signal_handlers_registered:
            loop.remove_signal_handler(signal.SIGTERM)
            loop.remove_signal_handler(signal.SIGINT)
            return
        if not fallback_handlers_registered:
            return
        if original_sigterm is not None:
            with contextlib.suppress(ValueError):
                signal.signal(signal.SIGTERM, original_sigterm)
        if original_sigint is not None:
            with contextlib.suppress(ValueError):
                signal.signal(signal.SIGINT, original_sigint)

    return cleanup


async def _run_worker_with_shutdown(worker: "Worker") -> None:
    import asyncio

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    cleanup_handlers = _register_shutdown_handlers(loop, shutdown_event)

    try:
        await worker.queue.connect()
        worker_task = asyncio.create_task(worker.start())
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            [worker_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if worker_task in done:
            worker_task.result()

        if shutdown_event.is_set():
            await worker.stop()
            if not worker_task.done():
                worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task
            pending.discard(worker_task)

        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    finally:
        await worker.queue.disconnect()
        cleanup_handlers()


def build_cli_app() -> "Group":  # noqa: C901, PLR0915
    import asyncio
    import multiprocessing
    import platform
    from typing import cast

    from click import IntRange, group, option
    from litestar.cli._utils import LitestarGroup, console  # pyright: ignore

    @group(cls=LitestarGroup, name="workers", no_args_is_help=True)
    def background_worker_group() -> None:
        """Manage background task workers."""

    @background_worker_group.command(
        name="run",
        help="Run background worker processes.",
    )
    @option(
        "--workers",
        help="The number of worker processes to start.",
        type=IntRange(min=1),
        default=1,
        required=False,
        show_default=True,
    )
    @option(
        "--queues",
        help="List of queue names to process.",
        type=str,
        multiple=True,
        required=False,
        show_default=False,
    )
    @option("-v", "--verbose", help="Enable verbose logging.", is_flag=True, default=None, type=bool, required=False)
    @option("-d", "--debug", help="Enable debugging.", is_flag=True, default=None, type=bool, required=False)
    def run_worker(  # pyright: ignore[reportUnusedFunction]
        app: "Litestar",
        workers: int,
        queues: "Optional[tuple[str, ...]]",
        verbose: "Optional[bool]",
        debug: "Optional[bool]",
    ) -> None:
        """Run the API server."""
        console.rule("[yellow]Starting SAQ Workers[/]", align="left")
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)

        if app.logging_config is not None:
            app.logging_config.configure()
        if debug is not None or verbose is not None:
            app.debug = True
        plugin = get_saq_plugin(app)
        if queues:
            queue_list = list(queues)
            limited_start_up(plugin, queue_list)
        show_saq_info(app, workers, plugin)
        managed_workers = list(plugin.get_workers().values())
        processes: list[multiprocessing.Process] = []
        shutdown_timeout = get_max_shutdown_timeout(managed_workers)

        def handle_shutdown_signal(signum: int, _frame: Any) -> None:
            """Handle shutdown signals (SIGTERM/SIGINT) for graceful shutdown."""
            sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
            console.print(f"[yellow]Received {sig_name}, stopping workers (timeout: {shutdown_timeout:.1f}s)...[/]")
            _terminate_worker_processes(processes, timeout=shutdown_timeout)
            loop = asyncio.get_event_loop()
            for w in managed_workers:
                loop.run_until_complete(w.stop())
            console.print("[yellow]SAQ workers stopped.[/]")
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_shutdown_signal)
        signal.signal(signal.SIGINT, handle_shutdown_signal)

        if platform.system() == "Darwin":
            ctx = multiprocessing.get_context("spawn")
        else:
            ctx = multiprocessing.get_context()
        use_spawn = ctx.get_start_method() == "spawn"
        app_path = plugin.config.app_path or os.environ.get("LITESTAR_APP") if use_spawn else None
        if use_spawn and not app_path:
            msg = (
                "SAQ worker spawn requires app_path in SAQConfig or LITESTAR_APP "
                "to be set (e.g. 'module:app' or 'module:create_app')."
            )
            raise RuntimeError(msg)

        if workers > 1:
            for _ in range(workers - 1):
                for worker in managed_workers:
                    if use_spawn:
                        p = ctx.Process(
                            target=run_saq_worker_from_app_path,
                            args=(cast(str, app_path), worker.queue.name),
                        )
                    else:
                        p = ctx.Process(
                            target=run_saq_worker,
                            args=(
                                worker,
                                app.logging_config,
                            ),
                        )
                    p.start()
                    processes.append(p)

        if len(managed_workers) > 1:
            for j in range(len(managed_workers) - 1):
                if use_spawn:
                    p = ctx.Process(
                        target=run_saq_worker_from_app_path,
                        args=(cast(str, app_path), managed_workers[j + 1].queue.name),
                    )
                else:
                    p = ctx.Process(target=run_saq_worker, args=(managed_workers[j + 1], app.logging_config))
                p.start()
                processes.append(p)

        run_saq_worker(
            worker=managed_workers[0],
            logging_config=cast("BaseLoggingConfig", app.logging_config),
        )

    @background_worker_group.command(
        name="status",
        help="Check the status of currently configured workers and queues.",
    )
    @option("-v", "--verbose", help="Enable verbose logging.", is_flag=True, default=None, type=bool, required=False)
    @option("-d", "--debug", help="Enable debugging.", is_flag=True, default=None, type=bool, required=False)
    def worker_status(  # pyright: ignore[reportUnusedFunction]
        app: "Litestar",
        verbose: "Optional[bool]",
        debug: "Optional[bool]",
    ) -> None:
        """Check the status of currently configured workers and queues."""
        console.rule("[yellow]Checking SAQ worker status[/]", align="left")
        if app.logging_config is not None:
            app.logging_config.configure()
        if debug is not None or verbose is not None:
            app.debug = True
        plugin = get_saq_plugin(app)
        show_saq_info(app, plugin.config.worker_processes, plugin)

    return background_worker_group


def limited_start_up(plugin: "SAQPlugin", queues: "list[str]") -> None:
    """Reset the workers and include only the specified queues."""
    plugin.remove_workers()
    plugin.config.filter_delete_queues(queues)


def get_saq_plugin(app: "Litestar") -> "SAQPlugin":
    """Retrieve a SAQ plugin from the Litestar application's plugins.

    This function attempts to find a SAQ plugin instance.
    If plugin is not found, it raises an ImproperlyConfiguredException.

    Args:
        app: The Litestar application instance.

    Returns:
        The SAQ plugin instance.

    Raises:
        ImproperConfigurationError: If the SAQ plugin is not found.
    """
    from contextlib import suppress

    from litestar_saq.exceptions import ImproperConfigurationError
    from litestar_saq.plugin import SAQPlugin

    with suppress(KeyError):
        return app.plugins.get(SAQPlugin)
    msg = "Failed to initialize SAQ. The required plugin (SAQPlugin) is missing."
    raise ImproperConfigurationError(
        msg,
    )


def show_saq_info(app: "Litestar", workers: int, plugin: "SAQPlugin") -> None:  # pragma: no cover
    """Display basic information about the application and its configuration."""

    from litestar.cli._utils import _format_is_enabled, console  # pyright: ignore
    from rich.table import Table
    from saq import __version__ as saq_version

    table = Table(show_header=False)
    table.add_column("title", style="cyan")
    table.add_column("value", style="bright_blue")

    table.add_row("SAQ version", saq_version)
    table.add_row("Debug mode", _format_is_enabled(app.debug))
    table.add_row("Number of Processes", str(workers))
    table.add_row("Queues", str(len(plugin.config.queue_configs)))

    console.print(table)


def run_saq_worker_from_app_path(app_path: str, worker_name: str) -> None:
    """Run a worker in a spawned process by reloading the Litestar app."""
    from litestar.cli._utils import _load_app_from_path

    loaded_app = _load_app_from_path(app_path)
    app = loaded_app.app
    plugin = get_saq_plugin(app)
    worker = plugin.get_workers()[worker_name]
    run_saq_worker(worker, app.logging_config)


def run_saq_worker(worker: "Worker", logging_config: "Optional[BaseLoggingConfig]") -> None:
    """Run a worker.

    Args:
        worker: The worker instance to run.
        logging_config: Optional logging configuration to apply.
    """
    from litestar.logging.config import StructLoggingConfig

    loop = _get_event_loop()

    if logging_config is not None:
        logging_config.configure()

    # Configure structlog context for separate process workers
    # (In-process workers configure in on_app_startup)
    if worker.separate_process:
        worker.configure_structlog_context()
        if isinstance(logging_config, StructLoggingConfig) and logging_config.standard_lib_logging_config is not None:
            logging_config.standard_lib_logging_config.configure()

    try:
        if worker.separate_process:
            loop.run_until_complete(_run_worker_with_shutdown(worker))
    except KeyboardInterrupt:
        loop.run_until_complete(loop.create_task(worker.stop()))
