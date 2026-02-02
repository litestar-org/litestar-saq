"""Heartbeat manager for batched job heartbeat updates.

This module provides a centralized heartbeat manager that batches and
deduplicates heartbeat updates across all jobs, using a single connection
per flush cycle instead of one per job.

Usage:
    >>> from litestar_saq.heartbeat import HeartbeatManager
    >>>
    >>> manager = HeartbeatManager(queue=queue, flush_interval=30.0)
    >>> manager.start()
    >>>
    >>> # Jobs signal the manager instead of calling job.update() directly
    >>> manager.register_job(job)
    >>> manager.signal(job.id)
    >>>
    >>> # On job completion
    >>> manager.unregister_job(job.id)
    >>>
    >>> # On worker shutdown
    >>> manager.stop()
"""

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from saq.job import Job
    from saq.queue.base import Queue

__all__ = ("HeartbeatManager",)

logger = logging.getLogger(__name__)


def _now() -> int:
    """Get current time in milliseconds (matches SAQ's now() function)."""
    import time

    return int(time.time() * 1000)


class HeartbeatManager:
    """Centralized heartbeat manager using batched updates.

    Instead of each job sending independent heartbeats, jobs signal
    this manager, which batches and deduplicates heartbeats for
    efficient single-connection updates.

    Thread Architecture:
        - Manager runs in a daemon thread with its own event loop
        - Jobs call signal(job_id) to request a heartbeat
        - Manager flushes pending heartbeats every flush_interval
        - Single connection acquired during flush, then released

    Backend Optimizations:
        - Redis: Uses pipeline for all SET operations in one round-trip
        - PostgreSQL: Uses single connection with transaction for all UPDATEs
        - Other: Falls back to sequential job.update() calls

    Args:
        queue: SAQ Queue instance (used for backend-specific batching).
        flush_interval: Seconds between batch flushes (default: 30.0).

    Example:
        Basic usage with worker lifecycle::

            manager = HeartbeatManager(queue=worker.queue, flush_interval=30.0)
            manager.start()

            # During job execution
            manager.register_job(job)
            manager.signal(job.id)  # Called periodically by decorator

            # When job completes
            manager.unregister_job(job.id)

            # On worker shutdown
            manager.stop()
    """

    def __init__(
        self,
        queue: "Queue",
        flush_interval: float = 30.0,
    ) -> None:
        self._queue = queue
        self._flush_interval = flush_interval
        self._pending: set[str] = set()  # job_ids awaiting heartbeat
        self._pending_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._jobs: dict[str, Job] = {}  # job_id -> Job reference
        self._jobs_lock = threading.Lock()

    def start(self) -> None:
        """Start the heartbeat manager thread.

        Creates a daemon thread that runs the flush loop. The thread
        has its own event loop, isolated from the job's event loop.

        This method is idempotent - calling it multiple times has no effect
        if the thread is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            return  # Already running

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="heartbeat-manager",
        )
        self._thread.start()
        logger.debug("HeartbeatManager started")

    def stop(self) -> None:
        """Stop the heartbeat manager gracefully.

        Signals the thread to stop and waits up to 5 seconds for it
        to terminate. After stopping, pending heartbeats are lost.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            logger.debug("HeartbeatManager stopped")

    def register_job(self, job: "Job") -> None:
        """Register a job for heartbeat management.

        Jobs must be registered before their signals can be processed.
        The job reference is stored so update() can be called during flush.

        Args:
            job: SAQ Job instance to register.
        """
        with self._jobs_lock:
            self._jobs[job.id] = job

    def unregister_job(self, job_id: str) -> None:
        """Unregister a job from heartbeat management.

        Called when a job completes, fails, or is cancelled. Removes
        the job reference and any pending signals for this job.

        Args:
            job_id: ID of the job to unregister.
        """
        with self._jobs_lock:
            self._jobs.pop(job_id, None)
        with self._pending_lock:
            self._pending.discard(job_id)

    def signal(self, job_id: str) -> None:
        """Signal that a job needs a heartbeat.

        Thread-safe and non-blocking. Multiple signals for the same job
        within the flush window are automatically deduplicated - only
        one heartbeat will be sent per job per flush.

        Args:
            job_id: ID of the job that needs a heartbeat.
        """
        with self._pending_lock:
            self._pending.add(job_id)

    def _run(self) -> None:
        """Thread entry point - creates isolated event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._manager_loop())
        except Exception:
            logger.exception("HeartbeatManager thread crashed")
        finally:
            self._loop.close()

    async def _manager_loop(self) -> None:
        """Main loop - flush pending heartbeats every interval."""
        while not self._stop_event.is_set():
            # Use wait with timeout for responsive shutdown
            await asyncio.sleep(self._flush_interval)
            if not self._stop_event.is_set():
                await self._flush()

    async def _flush(self) -> None:
        """Flush all pending heartbeats in a single batch.

        Atomically drains the pending set, looks up job references,
        and updates all jobs using backend-specific batching.
        """
        # Atomically drain pending set
        with self._pending_lock:
            if not self._pending:
                return
            pending_ids = self._pending.copy()
            self._pending.clear()

        # Get job references
        with self._jobs_lock:
            jobs_to_update = [self._jobs[job_id] for job_id in pending_ids if job_id in self._jobs]

        if not jobs_to_update:
            return

        # Batch update using backend-specific optimization
        try:
            await self._batch_update(jobs_to_update)
            logger.debug(
                "Heartbeat batch sent for %d jobs",
                len(jobs_to_update),
            )
        except Exception:
            logger.exception("Heartbeat batch failed")

    async def _batch_update(self, jobs: list["Job"]) -> None:
        """Perform batched heartbeat update with backend-specific optimization.

        Detects the queue backend and uses the most efficient batching:
        - Redis: Pipeline all SET operations in one round-trip
        - PostgreSQL: Single transaction with all UPDATEs
        - Other: Sequential job.update() calls (still benefits from dedup)

        Args:
            jobs: List of jobs to update.
        """
        # Import here to avoid circular imports and allow optional dependencies
        try:
            from saq.queue.postgres import PostgresQueue
        except ImportError:
            PostgresQueue = None  # type: ignore[misc,assignment]

        try:
            from saq.queue.redis import RedisQueue
        except ImportError:
            RedisQueue = None  # type: ignore[misc,assignment]

        # Dispatch to backend-specific implementation
        if RedisQueue is not None and isinstance(self._queue, RedisQueue):
            await self._batch_update_redis(jobs)
        elif PostgresQueue is not None and isinstance(self._queue, PostgresQueue):
            await self._batch_update_postgres(jobs)
        else:
            await self._batch_update_fallback(jobs)

    async def _batch_update_redis(self, jobs: list["Job"]) -> None:
        """Batch update using Redis pipeline - single round-trip for all jobs.

        Args:
            jobs: List of jobs to update.
        """
        queue = self._queue
        now_ts = _now()

        # Update touched timestamp on all jobs
        for job in jobs:
            job.touched = now_ts

        # Pipeline all SET operations
        try:
            async with queue.redis.pipeline(transaction=True) as pipe:  # type: ignore[attr-defined]
                for job in jobs:
                    pipe.set(job.id, queue.serialize(job))
                await pipe.execute()

            # Notify subscribers (required for SAQ's job tracking)
            for job in jobs:
                try:
                    await queue.notify(job)  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    # Notification failure shouldn't fail the heartbeat
                    logger.debug("Failed to notify for job %s", job.id)

        except Exception:  # noqa: BLE001
            logger.warning("Redis pipeline failed, falling back to sequential updates", exc_info=True)
            await self._batch_update_fallback(jobs)

    async def _batch_update_postgres(self, jobs: list["Job"]) -> None:
        """Batch update using PostgreSQL single transaction.

        Args:
            jobs: List of jobs to update.
        """
        try:
            from psycopg.sql import SQL
        except ImportError:
            logger.warning("psycopg not available, falling back to sequential updates")
            await self._batch_update_fallback(jobs)
            return

        queue = self._queue
        now_ts = _now()

        # Update touched timestamp on all jobs
        for job in jobs:
            job.touched = now_ts

        try:
            # Single connection, single transaction for all updates
            async with queue.pool.connection() as conn, conn.transaction():  # type: ignore[attr-defined]
                for job in jobs:
                    await conn.execute(
                        SQL(
                            """
                            UPDATE {jobs_table} SET
                                job = %(job)s,
                                status = %(status)s,
                                scheduled = %(scheduled)s
                            WHERE key = %(key)s
                            """
                        ).format(jobs_table=queue.jobs_table),  # type: ignore[attr-defined]
                        {
                            "job": queue.serialize(job),
                            "status": job.status,
                            "key": job.key,
                            "scheduled": job.scheduled,
                        },
                    )
        except Exception:  # noqa: BLE001
            logger.warning("PostgreSQL transaction failed, falling back to sequential updates", exc_info=True)
            await self._batch_update_fallback(jobs)

    async def _batch_update_fallback(self, jobs: list["Job"]) -> None:
        """Fallback: Sequential job.update() calls.

        Used when backend-specific batching is not available or fails.
        Still benefits from deduplication.

        Args:
            jobs: List of jobs to update.
        """
        for job in jobs:
            try:
                await job.update()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to update job %s",
                    job.id,
                    exc_info=True,
                )
