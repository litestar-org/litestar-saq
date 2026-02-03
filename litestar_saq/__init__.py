from __future__ import annotations

from litestar_saq.base import CronJob, Job, Worker
from litestar_saq.config import PostgresQueueOptions, QueueConfig, RedisQueueOptions, SAQConfig, TaskQueues
from litestar_saq.decorators import monitored_job
from litestar_saq.heartbeat import HeartbeatManager
from litestar_saq.hooks import (
    after_process_logger,
    before_process_logger,
    shutdown_logger,
    startup_logger,
    timing_after_process,
    timing_before_process,
)
from litestar_saq.plugin import SAQPlugin
from litestar_saq.typing import OPENTELEMETRY_INSTALLED

__all__ = (
    # OpenTelemetry
    "OPENTELEMETRY_INSTALLED",
    "CronJob",
    # Heartbeat
    "HeartbeatManager",
    "Job",
    "PostgresQueueOptions",
    "QueueConfig",
    "RedisQueueOptions",
    "SAQConfig",
    "SAQPlugin",
    "TaskQueues",
    "Worker",
    # Default hooks
    "after_process_logger",
    "before_process_logger",
    # Decorators
    "monitored_job",
    "shutdown_logger",
    "startup_logger",
    "timing_after_process",
    "timing_before_process",
)
