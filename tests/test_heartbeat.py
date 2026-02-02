"""Tests for HeartbeatManager."""

import threading
import time
from unittest.mock import AsyncMock, Mock

import pytest

from litestar_saq.heartbeat import HeartbeatManager

pytestmark = pytest.mark.anyio


# =============================================================================
# Signal Deduplication Tests
# =============================================================================


def test_signal_adds_to_pending() -> None:
    """Test that signal() adds job_id to pending set."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    manager.signal("job-123")

    assert "job-123" in manager._pending


def test_signal_deduplication() -> None:
    """Test that multiple signals for same job result in one pending entry."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    # Signal same job 3 times
    manager.signal("job-123")
    manager.signal("job-123")
    manager.signal("job-123")

    # Should only have 1 entry
    assert len(manager._pending) == 1
    assert "job-123" in manager._pending


def test_signal_multiple_jobs() -> None:
    """Test that signals for different jobs are all tracked."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    manager.signal("job-1")
    manager.signal("job-2")
    manager.signal("job-3")

    assert len(manager._pending) == 3
    assert manager._pending == {"job-1", "job-2", "job-3"}


def test_signal_is_thread_safe() -> None:
    """Test that signal() can be called from multiple threads safely."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    def signal_from_thread(job_id: str) -> None:
        for _ in range(100):
            manager.signal(job_id)

    threads = [threading.Thread(target=signal_from_thread, args=(f"job-{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have 10 unique job_ids
    assert len(manager._pending) == 10


# =============================================================================
# Job Registration Tests
# =============================================================================


def test_register_job() -> None:
    """Test that register_job() stores job reference."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    job_mock = Mock()
    job_mock.id = "job-123"

    manager.register_job(job_mock)

    assert "job-123" in manager._jobs
    assert manager._jobs["job-123"] is job_mock


def test_unregister_job() -> None:
    """Test that unregister_job() removes job and pending signal."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    job_mock = Mock()
    job_mock.id = "job-123"

    manager.register_job(job_mock)
    manager.signal("job-123")

    manager.unregister_job("job-123")

    assert "job-123" not in manager._jobs
    assert "job-123" not in manager._pending


def test_unregister_nonexistent_job() -> None:
    """Test that unregister_job() handles missing job gracefully."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    # Should not raise
    manager.unregister_job("nonexistent")


# =============================================================================
# Thread Lifecycle Tests
# =============================================================================


def test_start_creates_daemon_thread() -> None:
    """Test that start() creates a daemon thread."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    manager.start()

    try:
        assert manager._thread is not None
        assert manager._thread.daemon is True
        assert manager._thread.name == "heartbeat-manager"
        assert manager._thread.is_alive()
    finally:
        manager.stop()


def test_start_is_idempotent() -> None:
    """Test that calling start() twice doesn't create a second thread."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    manager.start()
    first_thread = manager._thread

    manager.start()  # Second call

    try:
        assert manager._thread is first_thread
    finally:
        manager.stop()


def test_stop_terminates_thread() -> None:
    """Test that stop() terminates the thread gracefully."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=0.1)

    manager.start()
    assert manager._thread is not None
    assert manager._thread.is_alive()

    manager.stop()

    assert not manager._thread.is_alive() if manager._thread else True


def test_stop_without_start() -> None:
    """Test that stop() handles case where thread was never started."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    # Should not raise
    manager.stop()


# =============================================================================
# Flush Behavior Tests
# =============================================================================


async def test_flush_clears_pending() -> None:
    """Test that flush() clears the pending set."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    # Register jobs so they can be found
    job1 = AsyncMock()
    job1.id = "job-1"
    job2 = AsyncMock()
    job2.id = "job-2"

    manager.register_job(job1)
    manager.register_job(job2)
    manager.signal("job-1")
    manager.signal("job-2")

    await manager._flush()

    assert len(manager._pending) == 0


async def test_flush_calls_job_update() -> None:
    """Test that flush() calls update() on each pending job."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    job1 = AsyncMock()
    job1.id = "job-1"
    job2 = AsyncMock()
    job2.id = "job-2"

    manager.register_job(job1)
    manager.register_job(job2)
    manager.signal("job-1")
    manager.signal("job-2")

    await manager._flush()

    job1.update.assert_called_once()
    job2.update.assert_called_once()


async def test_flush_skips_unregistered_jobs() -> None:
    """Test that flush() ignores signals for jobs no longer registered."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    job1 = AsyncMock()
    job1.id = "job-1"

    manager.register_job(job1)
    manager.signal("job-1")
    manager.signal("job-ghost")  # Not registered

    await manager._flush()

    # Only job-1 should have been updated
    job1.update.assert_called_once()


async def test_flush_continues_on_update_error() -> None:
    """Test that flush() continues updating other jobs if one fails."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    job1 = AsyncMock()
    job1.id = "job-1"
    job1.update.side_effect = RuntimeError("Connection lost")

    job2 = AsyncMock()
    job2.id = "job-2"

    manager.register_job(job1)
    manager.register_job(job2)
    manager.signal("job-1")
    manager.signal("job-2")

    # Should not raise
    await manager._flush()

    # job2 should still have been updated
    job2.update.assert_called_once()


async def test_flush_no_op_when_pending_empty() -> None:
    """Test that flush() does nothing when no signals pending."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=30.0)

    # No signals, just flush
    await manager._flush()

    # Should complete without error


# =============================================================================
# Integration Tests
# =============================================================================


def test_manager_thread_runs_flush_loop() -> None:
    """Test that the manager thread periodically calls flush."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=0.05)  # 50ms

    job1 = AsyncMock()
    job1.id = "job-1"

    manager.register_job(job1)
    manager.signal("job-1")

    manager.start()

    try:
        # Wait for at least one flush cycle
        time.sleep(0.15)

        # Job should have been updated
        assert job1.update.call_count >= 1
    finally:
        manager.stop()


def test_manager_deduplicates_across_flush() -> None:
    """Test that multiple signals between flushes result in one update."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=0.1)

    job1 = AsyncMock()
    job1.id = "job-1"
    manager.register_job(job1)

    manager.start()

    try:
        # Signal multiple times rapidly
        for _ in range(10):
            manager.signal("job-1")
            time.sleep(0.01)

        # Wait for flush
        time.sleep(0.15)

        # Should have called update only once per flush cycle
        # (might be 1 or 2 depending on timing, but definitely not 10)
        assert job1.update.call_count <= 2
    finally:
        manager.stop()


# =============================================================================
# Configuration Tests
# =============================================================================


def test_default_flush_interval() -> None:
    """Test that default flush interval is 30 seconds."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock)

    assert manager._flush_interval == 30.0


def test_custom_flush_interval() -> None:
    """Test that flush interval can be customized."""
    queue_mock = Mock()
    manager = HeartbeatManager(queue=queue_mock, flush_interval=60.0)

    assert manager._flush_interval == 60.0


# =============================================================================
# Import Tests
# =============================================================================


def test_heartbeat_manager_importable() -> None:
    """Test HeartbeatManager can be imported from heartbeat module."""
    from litestar_saq.heartbeat import HeartbeatManager as HM

    assert HM is not None


def test_heartbeat_manager_in_all() -> None:
    """Test HeartbeatManager is in __all__."""
    from litestar_saq import heartbeat

    assert "HeartbeatManager" in heartbeat.__all__


# =============================================================================
# Worker Integration Tests
# =============================================================================


def test_worker_creates_heartbeat_manager_when_enabled() -> None:
    """Test Worker creates HeartbeatManager when enable_heartbeat_manager=True."""
    from litestar_saq.base import Worker

    queue_mock = Mock()

    async def dummy_task(ctx: dict) -> None:
        pass

    worker = Worker(
        queue=queue_mock,
        functions=[dummy_task],
        enable_heartbeat_manager=True,
        heartbeat_flush_interval=15.0,
    )

    assert worker._enable_heartbeat_manager is True
    assert worker._heartbeat_flush_interval == 15.0
    # Manager is created lazily in on_app_startup, so it's None initially
    assert worker._heartbeat_manager is None


def test_worker_creates_manager_by_default() -> None:
    """Test Worker creates HeartbeatManager by default (enabled=True)."""
    from litestar_saq.base import Worker

    queue_mock = Mock()

    async def dummy_task(ctx: dict) -> None:
        pass

    worker = Worker(
        queue=queue_mock,
        functions=[dummy_task],
    )

    # Default is now True
    assert worker._enable_heartbeat_manager is True
    # Manager is created lazily in on_app_startup, so it's None initially
    assert worker._heartbeat_manager is None


def test_worker_does_not_create_manager_when_explicitly_disabled() -> None:
    """Test Worker doesn't create HeartbeatManager when explicitly disabled."""
    from litestar_saq.base import Worker

    queue_mock = Mock()

    async def dummy_task(ctx: dict) -> None:
        pass

    worker = Worker(
        queue=queue_mock,
        functions=[dummy_task],
        enable_heartbeat_manager=False,
    )

    assert worker._enable_heartbeat_manager is False
    assert worker._heartbeat_manager is None


@pytest.mark.anyio
async def test_worker_starts_heartbeat_manager_on_startup() -> None:
    """Test Worker starts HeartbeatManager in on_app_startup."""
    from litestar_saq.base import Worker

    queue_mock = Mock()

    async def dummy_task(ctx: dict) -> None:
        pass

    worker = Worker(
        queue=queue_mock,
        functions=[dummy_task],
        enable_heartbeat_manager=True,
        separate_process=True,  # Prevents actual task creation
    )

    await worker.on_app_startup()

    try:
        assert worker._heartbeat_manager is not None
        assert worker._heartbeat_manager._thread is not None
        assert worker._heartbeat_manager._thread.is_alive()
    finally:
        await worker.on_app_shutdown()


@pytest.mark.anyio
async def test_worker_stops_heartbeat_manager_on_shutdown() -> None:
    """Test Worker stops HeartbeatManager in on_app_shutdown."""
    from litestar_saq.base import Worker

    queue_mock = Mock()

    async def dummy_task(ctx: dict) -> None:
        pass

    worker = Worker(
        queue=queue_mock,
        functions=[dummy_task],
        enable_heartbeat_manager=True,
        separate_process=True,
    )

    await worker.on_app_startup()
    manager = worker._heartbeat_manager
    assert manager is not None

    await worker.on_app_shutdown()

    assert worker._heartbeat_manager is None
    # Thread should have stopped
    time.sleep(0.1)  # Allow thread cleanup
    assert not manager._thread.is_alive()


@pytest.mark.anyio
async def test_worker_injects_manager_into_context() -> None:
    """Test Worker's before_process hook injects HeartbeatManager into context."""
    from litestar_saq.base import Worker

    queue_mock = Mock()

    async def dummy_task(ctx: dict) -> None:
        pass

    worker = Worker(
        queue=queue_mock,
        functions=[dummy_task],
        enable_heartbeat_manager=True,
        separate_process=True,
    )

    await worker.on_app_startup()

    try:
        # Simulate context dict
        ctx: dict = {}

        # Call the injection hook directly
        await worker._inject_heartbeat_manager(ctx)

        assert "heartbeat_manager" in ctx
        assert ctx["heartbeat_manager"] is worker._heartbeat_manager
    finally:
        await worker.on_app_shutdown()
