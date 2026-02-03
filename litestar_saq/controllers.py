import html
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, cast

from litestar.exceptions import NotFoundException

if TYPE_CHECKING:
    from litestar import Controller
    from litestar.types.callable_types import Guard  # pyright: ignore[reportUnknownVariableType]
    from saq.queue.base import Queue
    from saq.types import QueueInfo

    from litestar_saq.base import Job
    from litestar_saq.config import TaskQueues


async def job_info(queue: "Queue", job_id: str) -> "Job":
    """Return a job

    Raises:
        NotFoundException: If the queue or job is not found.
    """
    job = await queue.job(job_id)
    if not job:
        msg = f"Could not find job ID {job_id}"
        raise NotFoundException(msg)
    return cast("Job", job)


def get_queue_or_404(task_queues: "TaskQueues", queue_id: str) -> "Queue":
    """Return a queue or raise NotFoundException."""
    queue = task_queues.queues.get(queue_id)
    if queue is None:
        msg = f"Could not find the {queue_id} queue"
        raise NotFoundException(msg)
    return queue


def build_controller(
    url_base: str = "/saq",
    controller_guards: "Optional[list[Guard]]" = None,  # pyright: ignore[reportUnknownParameterType]
    include_in_schema_: bool = False,
) -> "type[Controller]":
    """Build SAQ controller with web UI endpoints.

    Args:
        url_base: Base URL path for the controller.
        controller_guards: Optional list of guards to apply to the controller.
        include_in_schema_: Whether to include routes in the OpenAPI schema.

    Returns:
        A configured SAQController class.
    """
    guards_tuple = tuple(controller_guards) if controller_guards else None  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]
    return _build_controller_cached(url_base, guards_tuple, include_in_schema_)  # pyright: ignore[reportUnknownArgumentType]


@lru_cache(typed=True)
def _build_controller_cached(
    url_base: str,
    controller_guards: "Optional[tuple[Guard, ...]]",  # pyright: ignore[reportUnknownParameterType]
    include_in_schema_: bool,
) -> "type[Controller]":
    """Internal cached controller builder.

    Args:
        url_base: Base URL path for the controller.
        controller_guards: Optional tuple of guards (hashable for caching).
        include_in_schema_: Whether to include routes in the OpenAPI schema.

    Returns:
        A configured SAQController class.
    """
    from litestar import Controller, MediaType, Response, get, post
    from litestar.status_codes import HTTP_202_ACCEPTED, HTTP_500_INTERNAL_SERVER_ERROR

    # Convert tuple back to list for the controller (Litestar expects list or None)
    guards_list = list(controller_guards) if controller_guards else None  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]

    normalized_root = url_base.rstrip("/") or "/saq"
    escaped_root = html.escape(normalized_root)
    html_template = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" type="text/css" href="{escaped_root}/static/pico.min.css.gz">
            <title>SAQ</title>
        </head>
        <body>
            <div id="app"></div>
            <script>const root_path = "{escaped_root}";</script>
            <script src="{escaped_root}/static/snabbdom.js.gz"></script>
            <script src="{escaped_root}/static/app.js"></script>
        </body>
    </html>
    """.strip()

    class SAQController(Controller):
        tags = ["SAQ"]
        guards = guards_list  # pyright: ignore[reportUnknownVariableType]
        include_in_schema = include_in_schema_

        @get(
            operation_id="WorkerQueueList",
            name="worker:queue-list",
            path=[f"{url_base}/api/queues"],
            media_type=MediaType.JSON,
            cache=False,
            summary="Queue List",
            description="List configured worker queues.",
        )
        async def queue_list(self, task_queues: "TaskQueues") -> "dict[str, list[QueueInfo]]":
            """Get Worker queues.

            Args:
                task_queues: The task queues.

            Returns:
                The worker queues.
            """
            return {"queues": [await queue.info() for queue in task_queues.queues.values()]}

        @get(
            operation_id="WorkerQueueDetail",
            name="worker:queue-detail",
            path=f"{url_base}/api/queues/{{queue_id:str}}",
            media_type=MediaType.JSON,
            cache=False,
            summary="Queue Detail",
            description="List queue details.",
        )
        async def queue_detail(self, task_queues: "TaskQueues", queue_id: str) -> "dict[str, QueueInfo]":
            """Get queue information.

            Args:
                task_queues: The task queues.
                queue_id: The queue ID.

            Raises:
                NotFoundException: If the queue is not found.

            Returns:
                The queue information.
            """
            queue = get_queue_or_404(task_queues, queue_id)
            return {"queue": await queue.info(jobs=True)}

        @get(
            operation_id="WorkerJobDetail",
            name="worker:job-detail",
            path=f"{url_base}/api/queues/{{queue_id:str}}/jobs/{{job_id:str}}",
            media_type=MediaType.JSON,
            cache=False,
            summary="Job Details",
            description="List job details.",
        )
        async def job_detail(
            self, task_queues: "TaskQueues", queue_id: str, job_id: str
        ) -> "dict[str, dict[str, Any]]":
            """Get job information.

            Args:
                task_queues: The task queues.
                queue_id: The queue ID.
                job_id: The job ID.

            Raises:
                NotFoundException: If the queue or job is not found.

            Returns:
                The job information.
            """
            queue = get_queue_or_404(task_queues, queue_id)
            job = await job_info(queue, job_id)
            job_dict = job.to_dict()
            if "kwargs" in job_dict:
                job_dict["kwargs"] = repr(job_dict["kwargs"])
            if "result" in job_dict:
                job_dict["result"] = repr(job_dict["result"])
            return {"job": job_dict}

        @post(
            operation_id="WorkerJobRetry",
            name="worker:job-retry",
            path=f"{url_base}/api/queues/{{queue_id:str}}/jobs/{{job_id:str}}/retry",
            media_type=MediaType.JSON,
            cache=False,
            summary="Job Retry",
            description="Retry a failed job..",
            status_code=HTTP_202_ACCEPTED,
        )
        async def job_retry(self, task_queues: "TaskQueues", queue_id: str, job_id: str) -> "dict[str, str]":
            """Retry job.

            Args:
                task_queues: The task queues.
                queue_id: The queue ID.
                job_id: The job ID.

            Returns:
                The job information.
            """
            queue = get_queue_or_404(task_queues, queue_id)
            job = await job_info(queue, job_id)
            await job.retry("retried from ui")
            return {}

        @post(
            operation_id="WorkerJobAbort",
            name="worker:job-abort",
            path=f"{url_base}/api/queues/{{queue_id:str}}/jobs/{{job_id:str}}/abort",
            media_type=MediaType.JSON,
            cache=False,
            summary="Job Abort",
            description="Abort active job.",
            status_code=HTTP_202_ACCEPTED,
        )
        async def job_abort(self, task_queues: "TaskQueues", queue_id: str, job_id: str) -> "dict[str, str]":
            """Abort job.

            Args:
                task_queues: The task queues.
                queue_id: The queue ID.
                job_id: The job ID.

            Raises:
                NotFoundException: If the queue or job is not found.

            Returns:
                The job information.
            """
            queue = get_queue_or_404(task_queues, queue_id)
            job = await job_info(queue, job_id)
            await job.abort("aborted from ui")
            return {}

        @get(
            operation_id="WorkerHealthCheck",
            name="worker:health",
            path=f"{url_base}/health",
            media_type=MediaType.TEXT,
            cache=False,
            summary="Health Check",
            description="Check if queues are accessible.",
        )
        async def health(self, task_queues: "TaskQueues") -> Response[str]:
            """Health check endpoint.

            Args:
                task_queues: The task queues.

            Returns:
                OK if queues are accessible, 500 otherwise.
            """
            try:
                queues_info = [await queue.info() for queue in task_queues.queues.values()]
                if queues_info:
                    return Response(content="OK", media_type=MediaType.TEXT)
            except Exception:  # noqa: BLE001, S110
                pass
            return Response(
                content="Service Unavailable",
                media_type=MediaType.TEXT,
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            )

        @get(
            [
                url_base,
                f"{url_base}/",
                f"{url_base}/queues/{{queue_id:str}}",
                f"{url_base}/queues/{{queue_id:str}}/jobs/{{job_id:str}}",
            ],
            operation_id="WorkerIndex",
            name="worker:index",
            media_type=MediaType.HTML,
            include_in_schema=False,
        )
        async def index(
            self,
            queue_id: Optional[str] = None,
            job_id: Optional[str] = None,
        ) -> str:
            return html_template

    return SAQController
