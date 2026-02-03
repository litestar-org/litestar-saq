"""Decorators for SAQ job functions.

This module provides decorators that add cross-cutting functionality to SAQ jobs,
such as automatic heartbeat monitoring.

Usage:
    >>> from litestar_saq import monitored_job
    >>>
    >>> @monitored_job(interval=10.0)
    >>> async def long_running_task(ctx):
    ...     # Long-running work with automatic heartbeats
    ...     await process_large_dataset()
    ...     return {"status": "complete"}
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TYPE_CHECKING, Optional, TypeVar, cast

from typing_extensions import Concatenate, ParamSpec

if TYPE_CHECKING:
    from saq.job import Job
    from saq.types import Context

    from litestar_saq.heartbeat import HeartbeatManager

__all__ = ("monitored_job",)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

# Minimum interval to prevent excessive overhead
MIN_HEARTBEAT_INTERVAL = 1.0

# Default interval when job has no heartbeat configured
DEFAULT_HEARTBEAT_INTERVAL = 5.0


def monitored_job(
    interval: Optional[float] = None,
) -> Callable[
    [Callable[Concatenate["Context", P], Awaitable[R]]],
    Callable[Concatenate["Context", P], Awaitable[R]],
]:
    """Decorator that adds automatic heartbeat monitoring to SAQ jobs.

    This decorator provides heartbeat monitoring to prevent long-running jobs
    from being marked as stuck by SAQ's heartbeat monitor. It signals the
    HeartbeatManager at the configured interval, which batches and deduplicates
    updates for efficient single-connection flushes.

    Args:
        interval: Seconds between heartbeat signals. If None (default), the interval
                 is automatically calculated as half of the job's heartbeat timeout,
                 with a minimum of 1 second. If the job has no heartbeat configured,
                 defaults to 5 seconds.

    Returns:
        Decorated function with automatic heartbeat monitoring.

    Raises:
        ValueError: If interval is explicitly set to <= 0.

    Example:
        Auto-calculated interval (recommended)::

            from litestar_saq import monitored_job

            @monitored_job()
            async def process_data(ctx):
                # Interval auto-calculated from job.heartbeat
                await process_large_dataset()
                return {"status": "complete"}

        Explicit interval override::

            @monitored_job(interval=30.0)
            async def train_model(ctx, model_id: str):
                # Override with explicit 30 second interval
                model = await load_model(model_id)
                for epoch in range(100):
                    await train_epoch(model)
                return {"model_id": model_id}

    Note:
        - Requires HeartbeatManager in context (auto-injected by Worker)
        - If no manager present, logs warning and runs without heartbeats
        - If no job in context, runs without heartbeats
        - Works with regular jobs, cron jobs, and scheduled jobs
        - Auto-calculated interval uses job.heartbeat / 2, floored at 1 second
    """
    if interval is not None and interval <= 0:
        msg = f"Heartbeat interval must be positive, got {interval}"
        raise ValueError(msg)

    def decorator(
        func: Callable[Concatenate["Context", P], Awaitable[R]],
    ) -> Callable[Concatenate["Context", P], Awaitable[R]]:
        @wraps(func)
        async def wrapper(ctx: "Context", *args: P.args, **kwargs: P.kwargs) -> R:
            job: Optional[Job] = ctx.get("job")  # pyright: ignore[reportUnknownMemberType]

            # If no job, just run the function
            if job is None:
                return await func(ctx, *args, **kwargs)

            # Check for HeartbeatManager in context
            manager: Optional[HeartbeatManager] = ctx.get("heartbeat_manager")  # type: ignore[assignment]  # pyright: ignore[reportUnknownMemberType]

            if manager is None:
                # No manager - log warning and run without heartbeats
                logger.warning(
                    "No HeartbeatManager in context for job %s - running without heartbeat monitoring. "
                    "Ensure Worker has enable_heartbeat_manager=True (default).",
                    getattr(job, "id", "unknown"),
                )
                return await func(ctx, *args, **kwargs)

            # Calculate effective interval
            effective_interval = _calculate_interval(job, interval)

            # Run with manager-based heartbeat signaling
            return await _run_with_manager(func, ctx, job, manager, effective_interval, *args, **kwargs)

        return cast("Callable[Concatenate[Context, P], Awaitable[R]]", wrapper)

    return decorator


async def _run_with_manager(
    func: Callable[Concatenate["Context", P], Awaitable[R]],
    ctx: "Context",
    job: "Job",
    manager: "HeartbeatManager",
    interval: float,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Run job with HeartbeatManager-based heartbeat signaling.

    Args:
        func: The job function to execute.
        ctx: SAQ context.
        job: The SAQ job.
        manager: HeartbeatManager instance.
        interval: Seconds between signals.
        *args: Positional arguments for func.
        **kwargs: Keyword arguments for func.

    Returns:
        Result from the job function.
    """
    job_id = getattr(job, "id", "unknown")

    # Register with manager
    manager.register_job(job)
    logger.debug(
        "Job %s registered with HeartbeatManager (signal interval: %.1fs)",
        job_id,
        interval,
    )

    # Start signaling loop
    signal_task = asyncio.create_task(_signal_loop(manager, job_id, interval))

    try:
        return await func(ctx, *args, **kwargs)
    finally:
        # Clean up signal task
        signal_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await signal_task

        # Unregister from manager
        manager.unregister_job(job_id)
        logger.debug("Job %s unregistered from HeartbeatManager", job_id)


async def _signal_loop(
    manager: "HeartbeatManager",
    job_id: str,
    interval: float,
) -> None:
    """Signal the HeartbeatManager at regular intervals.

    This runs as an asyncio task, periodically calling manager.signal()
    to indicate the job is still alive. The manager batches and deduplicates
    these signals.

    Args:
        manager: HeartbeatManager to signal.
        job_id: ID of the job to signal for.
        interval: Seconds between signals.
    """
    try:
        while True:
            await asyncio.sleep(interval)
            manager.signal(job_id)
    except asyncio.CancelledError:
        pass  # Expected on job completion


def _calculate_interval(job: "Optional[Job]", explicit_interval: Optional[float]) -> float:
    """Calculate the effective heartbeat interval.

    Args:
        job: The SAQ job, which may have a heartbeat setting.
        explicit_interval: Explicitly provided interval, or None for auto-calculation.

    Returns:
        The effective interval in seconds.
    """
    # If explicitly set, use that value
    if explicit_interval is not None:
        return explicit_interval

    # Auto-calculate from job's heartbeat setting
    if job is not None:
        job_heartbeat = getattr(job, "heartbeat", 0)
        if job_heartbeat > 0:
            # Use half the heartbeat timeout, with a floor of MIN_HEARTBEAT_INTERVAL
            calculated = job_heartbeat / 2
            return max(calculated, MIN_HEARTBEAT_INTERVAL)

    # Fallback to default
    return DEFAULT_HEARTBEAT_INTERVAL
