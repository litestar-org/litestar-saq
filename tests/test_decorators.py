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
# Heartbeat Signaling Tests (via HeartbeatManager)
# =============================================================================


async def test_monitored_job_signals_manager() -> None:
    """Test heartbeats are signaled to manager at the configured interval."""
    job_mock = AsyncMock()
    job_mock.id = "job-123"

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

    @monitored_job(interval=0.05)  # 50ms interval
    async def slow_task(ctx: Context) -> str:
        await asyncio.sleep(0.15)  # Run for 150ms
        return "done"

    result = await slow_task(ctx)

    assert result == "done"
    # Should have signaled at least 2 times (at 50ms and 100ms)
    assert manager_mock.signal.call_count >= 2


async def test_monitored_job_first_signal_after_interval() -> None:
    """Test first signal is sent after first interval, not immediately."""
    job_mock = AsyncMock()
    job_mock.id = "job-123"

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    signal_sent = False

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

    @monitored_job(interval=0.1)  # 100ms interval
    async def quick_task(ctx: Context) -> str:
        nonlocal signal_sent
        # This check happens immediately, before first signal
        signal_sent = manager_mock.signal.called
        return "done"

    await quick_task(ctx)

    # No signal should have been sent during the task because it completed
    # before the first interval elapsed
    assert not signal_sent


# =============================================================================
# Cleanup Tests
# =============================================================================


async def test_monitored_job_unregisters_on_completion() -> None:
    """Test job is unregistered when job completes successfully."""
    job_mock = AsyncMock()
    job_mock.id = "job-123"

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

    @monitored_job(interval=0.01)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.05)
        return "success"

    result = await task(ctx)

    assert result == "success"
    manager_mock.register_job.assert_called_once_with(job_mock)
    manager_mock.unregister_job.assert_called_once_with("job-123")


async def test_monitored_job_unregisters_on_exception() -> None:
    """Test job is unregistered when job raises exception."""
    job_mock = AsyncMock()
    job_mock.id = "job-456"

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

    @monitored_job(interval=0.01)
    async def failing_task(ctx: Context) -> None:
        await asyncio.sleep(0.03)
        msg = "Task failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="Task failed"):
        await failing_task(ctx)

    # Should still have unregistered
    manager_mock.unregister_job.assert_called_once_with("job-456")


async def test_monitored_job_stops_on_cancellation() -> None:
    """Test signal task is cancelled when job is cancelled externally."""
    job_mock = AsyncMock()
    job_mock.id = "job-789"

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

    @monitored_job(interval=0.01)
    async def long_task(ctx: Context) -> None:
        await asyncio.sleep(10)  # Would run for 10 seconds

    task = asyncio.create_task(long_task(ctx))  # pyright: ignore
    await asyncio.sleep(0.05)  # Let it run briefly
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


# =============================================================================
# Missing Job/Manager Context Tests
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


async def test_monitored_job_warns_without_manager(caplog: pytest.LogCaptureFixture) -> None:
    """Test warning logged when no HeartbeatManager in context."""
    job_mock = AsyncMock()
    job_mock.id = "job-no-manager"

    # No heartbeat_manager in context
    ctx = cast(Context, {"job": job_mock})

    @monitored_job(interval=0.02)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.01)
        return "done"

    with caplog.at_level(logging.WARNING, logger="litestar_saq.decorators"):
        result = await task(ctx)

    assert result == "done"
    assert "No HeartbeatManager in context" in caplog.text
    assert "job-no-manager" in caplog.text


async def test_monitored_job_runs_without_heartbeat_when_no_manager() -> None:
    """Test job runs without heartbeats when no manager present."""
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
    # job.update should NOT have been called (no legacy fallback)
    assert job_mock.update.call_count == 0


# =============================================================================
# Return Value and Exception Propagation Tests
# =============================================================================


async def test_monitored_job_preserves_return_value() -> None:
    """Test job return value is preserved."""
    job_mock = AsyncMock()
    job_mock.id = "job-return"

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

    @monitored_job(interval=0.01)
    async def task(ctx: Context) -> dict[str, Any]:
        return {"status": "complete", "count": 42}

    result = await task(ctx)

    assert result == {"status": "complete", "count": 42}


async def test_monitored_job_preserves_exception() -> None:
    """Test job exception is propagated correctly."""
    job_mock = AsyncMock()
    job_mock.id = "job-exception"

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

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

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

    @monitored_job(interval=0.01)  # 10ms interval
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.05)  # 50ms runtime
        return "done"

    await task(ctx)

    # Should have signaled ~4 times
    assert manager_mock.signal.call_count >= 3


async def test_monitored_job_custom_long_interval() -> None:
    """Test long interval doesn't send signal for short tasks."""
    job_mock = AsyncMock()
    job_mock.id = "job-long-interval"

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

    @monitored_job(interval=1.0)  # 1 second interval
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.05)  # 50ms runtime
        return "done"

    await task(ctx)

    # No signals should have been sent (task shorter than interval)
    assert manager_mock.signal.call_count == 0


# =============================================================================
# Concurrent Jobs Tests
# =============================================================================


async def test_monitored_job_concurrent_execution() -> None:
    """Test multiple decorated jobs don't interfere with each other."""
    managers = [
        Mock(register_job=Mock(), unregister_job=Mock(), signal=Mock()),
        Mock(register_job=Mock(), unregister_job=Mock(), signal=Mock()),
        Mock(register_job=Mock(), unregister_job=Mock(), signal=Mock()),
    ]
    job_mocks = [
        AsyncMock(id="job-1"),
        AsyncMock(id="job-2"),
        AsyncMock(id="job-3"),
    ]

    results: list[str] = []

    @monitored_job(interval=0.01)
    async def task(ctx: Context, task_id: str) -> str:
        await asyncio.sleep(0.05)
        results.append(task_id)
        return task_id

    # Run tasks concurrently
    await asyncio.gather(
        task(cast(Context, {"job": job_mocks[0], "heartbeat_manager": managers[0]}), task_id="task-1"),
        task(cast(Context, {"job": job_mocks[1], "heartbeat_manager": managers[1]}), task_id="task-2"),
        task(cast(Context, {"job": job_mocks[2], "heartbeat_manager": managers[2]}), task_id="task-3"),
    )

    assert len(results) == 3
    assert set(results) == {"task-1", "task-2", "task-3"}

    # Each manager should have received signals
    for manager in managers:
        assert manager.signal.call_count >= 1


# =============================================================================
# Logging Tests
# =============================================================================


async def test_monitored_job_logs_registration(caplog: pytest.LogCaptureFixture) -> None:
    """Test job registration is logged at DEBUG level."""
    job_mock = AsyncMock()
    job_mock.id = "job-log-register"

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

    @monitored_job(interval=0.02)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.01)
        return "done"

    with caplog.at_level(logging.DEBUG, logger="litestar_saq.decorators"):
        await task(ctx)

    assert "job-log-register registered with HeartbeatManager" in caplog.text


async def test_monitored_job_logs_unregistration(caplog: pytest.LogCaptureFixture) -> None:
    """Test job unregistration is logged at DEBUG level."""
    job_mock = AsyncMock()
    job_mock.id = "job-log-unregister"

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

    @monitored_job(interval=0.01)
    async def task(ctx: Context) -> str:
        await asyncio.sleep(0.02)
        return "done"

    with caplog.at_level(logging.DEBUG, logger="litestar_saq.decorators"):
        await task(ctx)

    assert "job-log-unregister unregistered from HeartbeatManager" in caplog.text


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

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

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
    """Test signal loop handles job without id attribute."""
    job_mock = Mock()  # Regular Mock, not AsyncMock
    # Don't set id attribute
    del job_mock.id

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
    # Should still work, using "unknown" as job_id


async def test_monitored_job_with_positional_args() -> None:
    """Test decorator works with positional arguments."""
    job_mock = AsyncMock()
    job_mock.id = "job-args"

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

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

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

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

    manager_mock = Mock()
    manager_mock.register_job = Mock()
    manager_mock.unregister_job = Mock()
    manager_mock.signal = Mock()

    ctx = cast(Context, {"job": job_mock, "heartbeat_manager": manager_mock})

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
