"""Tests for SAQ job decorators."""

import asyncio
import logging
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from saq.types import Context

from litestar_saq.decorators import monitored_job

pytestmark = pytest.mark.anyio


# =============================================================================
# Metadata Preservation Tests
# =============================================================================


def test_monitored_job_preserves_name() -> None:
    """Test decorator preserves function __name__."""

    @monitored_job()
    async def my_task(ctx: Context) -> None:
        pass

    assert my_task.__name__ == "my_task"


def test_monitored_job_preserves_doc() -> None:
    """Test decorator preserves function __doc__."""

    @monitored_job()
    async def documented_task(ctx: Context) -> None:
        """This is my docstring."""
        pass

    assert documented_task.__doc__ == "This is my docstring."


def test_monitored_job_preserves_module() -> None:
    """Test decorator preserves function __module__."""

    @monitored_job()
    async def my_task(ctx: Context) -> None:
        pass

    assert my_task.__module__ == __name__


# =============================================================================
# Interval Validation Tests
# =============================================================================


def test_monitored_job_validates_zero_interval() -> None:
    """Test ValueError raised for zero interval."""
    with pytest.raises(ValueError, match="Heartbeat interval must be positive"):

        @monitored_job(interval=0)
        async def task(ctx: Context) -> None:
            pass


def test_monitored_job_validates_negative_interval() -> None:
    """Test ValueError raised for negative interval."""
    with pytest.raises(ValueError, match="Heartbeat interval must be positive"):

        @monitored_job(interval=-1.0)
        async def task(ctx: Context) -> None:
            pass


def test_monitored_job_accepts_positive_interval() -> None:
    """Test positive interval is accepted."""

    @monitored_job(interval=0.001)
    async def task(ctx: Context) -> None:
        pass

    # No exception raised
    assert callable(task)


def test_monitored_job_default_interval() -> None:
    """Test default interval is 5.0 seconds."""

    # This is implicit - if the decorator works without arguments, default is used
    @monitored_job()
    async def task(ctx: Context) -> None:
        pass

    assert callable(task)


# =============================================================================
# Heartbeat Sending Tests
# =============================================================================


async def test_monitored_job_sends_heartbeats() -> None:
    """Test heartbeats are sent at the configured interval."""
    job_mock = AsyncMock()
    job_mock.id = "job-123"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.05)  # 50ms interval
    async def slow_task(ctx: Context) -> str:
        await asyncio.sleep(0.15)  # Run for 150ms
        return "done"

    result = await slow_task(ctx)

    assert result == "done"
    # Should have sent at least 2 heartbeats (at 50ms and 100ms)
    assert job_mock.update.call_count >= 2


async def test_monitored_job_first_heartbeat_after_interval() -> None:
    """Test first heartbeat is sent after first interval, not immediately."""
    job_mock = AsyncMock()
    job_mock.id = "job-123"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    heartbeat_sent = False

    @monitored_job(interval=0.1)  # 100ms interval
    async def quick_task(ctx: Context) -> str:
        nonlocal heartbeat_sent
        # This check happens immediately, before first heartbeat
        heartbeat_sent = job_mock.update.called
        return "done"

    await quick_task(ctx)

    # No heartbeat should have been sent during the task because it completed
    # before the first interval elapsed
    assert not heartbeat_sent


# =============================================================================
# Cleanup Tests
# =============================================================================


async def test_monitored_job_stops_on_completion() -> None:
    """Test heartbeat task is cancelled when job completes successfully."""
    job_mock = AsyncMock()
    job_mock.id = "job-123"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.01)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.05)
        return "success"

    result = await task(ctx)

    assert result == "success"
    # After task completes, heartbeat should stop - give it a moment
    await asyncio.sleep(0.05)
    call_count_after = job_mock.update.call_count

    # Wait a bit more to ensure no more heartbeats are sent
    await asyncio.sleep(0.05)
    assert job_mock.update.call_count == call_count_after


async def test_monitored_job_stops_on_exception() -> None:
    """Test heartbeat task is cancelled when job raises exception."""
    job_mock = AsyncMock()
    job_mock.id = "job-456"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.01)
    async def failing_task(ctx: Context) -> None:
        await asyncio.sleep(0.03)
        msg = "Task failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="Task failed"):
        await failing_task(ctx)

    # After exception, heartbeat should stop
    await asyncio.sleep(0.05)
    call_count_after = job_mock.update.call_count

    # Wait a bit more to ensure no more heartbeats are sent
    await asyncio.sleep(0.05)
    assert job_mock.update.call_count == call_count_after


async def test_monitored_job_stops_on_cancellation() -> None:
    """Test heartbeat task is cancelled when job is cancelled externally."""
    job_mock = AsyncMock()
    job_mock.id = "job-789"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.01)
    async def long_task(ctx: Context) -> None:
        await asyncio.sleep(10)  # Would run for 10 seconds

    task = asyncio.create_task(long_task(ctx))  # pyright: ignore
    await asyncio.sleep(0.05)  # Let it run briefly
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


# =============================================================================
# Missing Job Context Tests
# =============================================================================


async def test_monitored_job_handles_missing_job() -> None:
    """Test graceful degradation when job not in context."""
    ctx = cast(Context, {})  # No job in context

    @monitored_job(interval=0.01)
    async def task(ctx: Context) -> str:
        return "completed"

    result = await task(ctx)

    assert result == "completed"


async def test_monitored_job_handles_none_job() -> None:
    """Test graceful degradation when job is explicitly None."""
    ctx = cast(Context, {"job": None})

    @monitored_job(interval=0.01)
    async def task(ctx: Context) -> str:
        return "completed"

    result = await task(ctx)

    assert result == "completed"


async def test_monitored_job_handles_missing_job_silently() -> None:
    """Test job runs without logging when job is missing from context.

    When no job is in context, heartbeat monitoring is silently skipped.
    The job function still executes normally without any errors or logs.
    """
    ctx = cast(Context, {})

    @monitored_job(interval=0.01)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.02)
        return "done"

    # Should complete without error
    result = await task(ctx)
    assert result == "done"


# =============================================================================
# Error Handling Tests
# =============================================================================


async def test_monitored_job_continues_on_heartbeat_error() -> None:
    """Test job continues when job.update() fails."""
    job_mock = AsyncMock()
    job_mock.id = "job-error"
    job_mock.update = AsyncMock(side_effect=RuntimeError("Connection lost"))
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.02)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.1)  # Long enough for multiple heartbeat attempts
        return "success"

    result = await task(ctx)

    # Job should complete successfully despite heartbeat failures
    assert result == "success"
    # Heartbeats should have been attempted
    assert job_mock.update.call_count >= 2


async def test_monitored_job_logs_heartbeat_error(caplog: pytest.LogCaptureFixture) -> None:
    """Test heartbeat failures are logged as warnings."""
    job_mock = AsyncMock()
    job_mock.id = "job-warning"
    job_mock.update = AsyncMock(side_effect=RuntimeError("Network error"))
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.02)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.05)
        return "done"

    with caplog.at_level(logging.WARNING, logger="litestar_saq.decorators"):
        await task(ctx)

    assert "Failed to send heartbeat" in caplog.text
    assert "job-warning" in caplog.text


async def test_monitored_job_continues_after_multiple_failures() -> None:
    """Test heartbeat loop continues after multiple consecutive failures."""
    call_count = 0

    async def failing_then_succeeding_update() -> None:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            msg = f"Failure {call_count}"
            raise RuntimeError(msg)
        # Succeed on third call

    job_mock = AsyncMock()
    job_mock.id = "job-retry"
    job_mock.update = AsyncMock(side_effect=failing_then_succeeding_update)
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.02)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.1)  # Long enough for multiple heartbeat attempts
        return "success"

    result = await task(ctx)

    assert result == "success"
    # At least 3 attempts should have been made
    assert call_count >= 3


# =============================================================================
# Return Value and Exception Propagation Tests
# =============================================================================


async def test_monitored_job_preserves_return_value() -> None:
    """Test job return value is preserved."""
    job_mock = AsyncMock()
    job_mock.id = "job-return"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.01)
    async def task(ctx: Context) -> dict[str, Any]:
        return {"status": "complete", "count": 42}

    result = await task(ctx)

    assert result == {"status": "complete", "count": 42}


async def test_monitored_job_preserves_exception() -> None:
    """Test job exception is propagated correctly."""
    job_mock = AsyncMock()
    job_mock.id = "job-exception"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    class CustomError(Exception):
        pass

    @monitored_job(interval=0.01)
    async def task(ctx: Context) -> None:
        raise CustomError("Something went wrong")

    with pytest.raises(CustomError, match="Something went wrong"):
        await task(ctx)


# =============================================================================
# Custom Interval Tests
# =============================================================================


async def test_monitored_job_custom_short_interval() -> None:
    """Test short custom interval."""
    job_mock = AsyncMock()
    job_mock.id = "job-short"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.01)  # 10ms interval
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.05)  # 50ms runtime
        return "done"

    await task(ctx)

    # Should have sent ~4 heartbeats
    assert job_mock.update.call_count >= 3


async def test_monitored_job_custom_long_interval() -> None:
    """Test long interval doesn't send heartbeat for short tasks."""
    job_mock = AsyncMock()
    job_mock.id = "job-long-interval"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=1.0)  # 1 second interval
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.05)  # 50ms runtime
        return "done"

    await task(ctx)

    # No heartbeats should have been sent (task shorter than interval)
    assert job_mock.update.call_count == 0


# =============================================================================
# Concurrent Jobs Tests
# =============================================================================


async def test_monitored_job_concurrent_execution() -> None:
    """Test multiple decorated jobs don't interfere with each other."""
    job_mocks = [
        AsyncMock(id="job-1", update=AsyncMock()),
        AsyncMock(id="job-2", update=AsyncMock()),
        AsyncMock(id="job-3", update=AsyncMock()),
    ]

    results: list[str] = []

    @monitored_job(interval=0.01)
    async def task(ctx: Context, task_id: str) -> str:
        await asyncio.sleep(0.05)
        results.append(task_id)
        return task_id

    # Run tasks concurrently
    await asyncio.gather(
        task(cast(Context, {"job": job_mocks[0]}), task_id="task-1"),
        task(cast(Context, {"job": job_mocks[1]}), task_id="task-2"),
        task(cast(Context, {"job": job_mocks[2]}), task_id="task-3"),
    )

    assert len(results) == 3
    assert set(results) == {"task-1", "task-2", "task-3"}

    # Each job should have received heartbeats
    for mock in job_mocks:
        assert mock.update.call_count >= 1


# =============================================================================
# Logging Tests
# =============================================================================


async def test_monitored_job_logs_heartbeat_sent(caplog: pytest.LogCaptureFixture) -> None:
    """Test successful heartbeats are logged at DEBUG level."""
    job_mock = AsyncMock()
    job_mock.id = "job-log-success"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.02)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.05)
        return "done"

    with caplog.at_level(logging.DEBUG, logger="litestar_saq.decorators"):
        await task(ctx)

    assert "Heartbeat sent for job job-log-success" in caplog.text


async def test_monitored_job_logs_monitoring_stopped(caplog: pytest.LogCaptureFixture) -> None:
    """Test monitoring stop is logged at DEBUG level."""
    job_mock = AsyncMock()
    job_mock.id = "job-log-stop"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.01)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.02)
        return "done"

    with caplog.at_level(logging.DEBUG, logger="litestar_saq.decorators"):
        await task(ctx)

    assert "Heartbeat monitoring stopped for job job-log-stop" in caplog.text


# =============================================================================
# Integration Tests
# =============================================================================


def test_monitored_job_importable_from_top_level() -> None:
    """Test monitored_job can be imported from litestar_saq package."""
    from litestar_saq import monitored_job as mj

    assert callable(mj)


def test_monitored_job_importable_from_decorators_module() -> None:
    """Test monitored_job can be imported from litestar_saq.decorators."""
    from litestar_saq.decorators import monitored_job as mj

    assert callable(mj)


def test_monitored_job_in_all() -> None:
    """Test monitored_job is in __all__."""
    from litestar_saq import decorators

    assert "monitored_job" in decorators.__all__


async def test_monitored_job_with_kwargs() -> None:
    """Test decorator works with keyword arguments."""
    job_mock = AsyncMock()
    job_mock.id = "job-kwargs"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.01)
    async def task(ctx: Context, name: str, count: int = 1) -> dict[str, Any]:
        return {"name": name, "count": count}

    result = await task(ctx, name="test", count=5)

    assert result == {"name": "test", "count": 5}


async def test_monitored_job_with_queue_config_tasks() -> None:
    """Test decorated function can be used in QueueConfig tasks."""
    from unittest.mock import MagicMock

    from litestar_saq import QueueConfig

    @monitored_job(interval=5.0)
    async def monitored_task(ctx: Context) -> None:
        pass

    broker = MagicMock()
    config = QueueConfig(
        broker_instance=broker,
        name="test-queue",
        tasks=[monitored_task],
    )

    assert config.tasks is not None
    assert len(config.tasks) == 1
    assert monitored_task in config.tasks


async def test_monitored_job_job_attribute_access() -> None:
    """Test heartbeat loop handles job without id attribute."""
    job_mock = Mock()  # Regular Mock, not AsyncMock
    job_mock.update = AsyncMock()
    # Don't set id attribute
    del job_mock.id
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.02)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.05)
        return "done"

    result = await task(ctx)

    assert result == "done"
    # Should still work, using "unknown" as job_id


async def test_monitored_job_with_positional_args() -> None:
    """Test decorator works with positional arguments."""
    job_mock = AsyncMock()
    job_mock.id = "job-args"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.01)
    async def task(ctx: Context, name: str, count: int) -> dict[str, Any]:
        return {"name": name, "count": count}

    # Pass positional arguments after ctx
    result = await task(ctx, "test-name", 42)

    assert result == {"name": "test-name", "count": 42}


async def test_monitored_job_with_mixed_args_kwargs() -> None:
    """Test decorator works with mixed positional and keyword arguments."""
    job_mock = AsyncMock()
    job_mock.id = "job-mixed"
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.01)
    async def task(ctx: Context, name: str, count: int, flag: bool = False) -> dict[str, Any]:
        return {"name": name, "count": count, "flag": flag}

    # Pass mixed positional and keyword arguments
    result = await task(ctx, "mixed", 10, flag=True)

    assert result == {"name": "mixed", "count": 10, "flag": True}


# =============================================================================
# Auto-Calculated Interval Tests
# =============================================================================


def test_calculate_interval_explicit_override() -> None:
    """Test explicit interval overrides auto-calculation."""
    from litestar_saq.decorators import _calculate_interval

    job_mock = Mock()
    job_mock.heartbeat = 60

    # Explicit interval should be used regardless of job.heartbeat
    result = _calculate_interval(job_mock, 10.0)
    assert result == 10.0


def test_calculate_interval_from_job_heartbeat() -> None:
    """Test interval auto-calculated as half of job.heartbeat."""
    from litestar_saq.decorators import _calculate_interval

    job_mock = Mock()
    job_mock.heartbeat = 60

    result = _calculate_interval(job_mock, None)
    assert result == 30.0  # 60 / 2


def test_calculate_interval_respects_floor() -> None:
    """Test interval respects minimum floor of 1 second."""
    from litestar_saq.decorators import _calculate_interval

    job_mock = Mock()
    job_mock.heartbeat = 1  # Would calculate to 0.5

    result = _calculate_interval(job_mock, None)
    assert result == 1.0  # Floored to MIN_HEARTBEAT_INTERVAL


def test_calculate_interval_no_job() -> None:
    """Test fallback to default when no job."""
    from litestar_saq.decorators import DEFAULT_HEARTBEAT_INTERVAL, _calculate_interval

    result = _calculate_interval(None, None)
    assert result == DEFAULT_HEARTBEAT_INTERVAL


def test_calculate_interval_job_heartbeat_zero() -> None:
    """Test fallback to default when job.heartbeat is 0 (disabled)."""
    from litestar_saq.decorators import DEFAULT_HEARTBEAT_INTERVAL, _calculate_interval

    job_mock = Mock()
    job_mock.heartbeat = 0  # Heartbeat disabled

    result = _calculate_interval(job_mock, None)
    assert result == DEFAULT_HEARTBEAT_INTERVAL


def test_calculate_interval_job_missing_heartbeat_attr() -> None:
    """Test fallback when job doesn't have heartbeat attribute."""
    from litestar_saq.decorators import DEFAULT_HEARTBEAT_INTERVAL, _calculate_interval

    job_mock = Mock(spec=[])  # No attributes

    result = _calculate_interval(job_mock, None)
    assert result == DEFAULT_HEARTBEAT_INTERVAL


async def test_monitored_job_auto_interval_from_heartbeat() -> None:
    """Test decorator auto-calculates interval from job.heartbeat."""
    job_mock = AsyncMock()
    job_mock.id = "job-auto"
    job_mock.heartbeat = 60  # Should result in 30 second interval
    job_mock.update = AsyncMock()
    ctx = cast(Context, {"job": job_mock})

    @monitored_job()  # No explicit interval
    async def task(ctx: Context) -> str:
        # Sleep long enough for heartbeat check but short for test
        await asyncio.sleep(0.01)
        return "done"

    result = await task(ctx)
    assert result == "done"


# =============================================================================
# HeartbeatManager Integration Tests
# =============================================================================


async def test_monitored_job_uses_manager_when_available() -> None:
    """Test decorator uses HeartbeatManager when present in context."""
    job_mock = AsyncMock()
    job_mock.id = "job-manager"
    job_mock.update = AsyncMock()

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

    @monitored_job(interval=0.02)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.05)
        return "done"

    result = await task(ctx)

    assert result == "done"
    # Manager should have been used
    manager_mock.register_job.assert_called_once_with(job_mock)
    manager_mock.unregister_job.assert_called_once_with("job-manager")
    # Signal should have been called at least once
    assert manager_mock.signal.call_count >= 1


async def test_monitored_job_signals_manager_periodically() -> None:
    """Test decorator signals manager at configured interval."""
    job_mock = AsyncMock()
    job_mock.id = "job-signal"
    job_mock.update = AsyncMock()

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

    @monitored_job(interval=0.02)  # 20ms interval
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.1)  # 100ms runtime
        return "done"

    await task(ctx)

    # Should have signaled multiple times (approximately 100/20 = 5 times)
    assert manager_mock.signal.call_count >= 3


async def test_monitored_job_unregisters_on_exception() -> None:
    """Test decorator unregisters job even when exception occurs."""
    job_mock = AsyncMock()
    job_mock.id = "job-exception"
    job_mock.update = AsyncMock()

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

    @monitored_job(interval=0.01)
    async def task(ctx: Context) -> None:
        await asyncio.sleep(0.02)
        msg = "Task failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="Task failed"):
        await task(ctx)

    # unregister should still be called
    manager_mock.unregister_job.assert_called_once_with("job-exception")


async def test_monitored_job_falls_back_without_manager() -> None:
    """Test decorator falls back to legacy behavior without manager."""
    job_mock = AsyncMock()
    job_mock.id = "job-legacy"
    job_mock.update = AsyncMock()

    # No heartbeat_manager in context
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.02)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.05)
        return "done"

    result = await task(ctx)

    assert result == "done"
    # Should have called job.update directly (legacy behavior)
    assert job_mock.update.call_count >= 1


async def test_monitored_job_manager_not_called_when_no_job() -> None:
    """Test decorator doesn't use manager when no job in context."""
    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"heartbeat_manager": manager_mock})

    @monitored_job(interval=0.01)
    async def task(ctx: Context) -> str:
        return "done"

    result = await task(ctx)

    assert result == "done"
    # Manager should not have been called (no job to register)
    manager_mock.register_job.assert_not_called()
    manager_mock.signal.assert_not_called()
