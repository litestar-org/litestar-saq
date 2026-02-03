from typing import Any

import pytest
from litestar import Litestar
from litestar.di import Provide
from litestar.exceptions import ImproperlyConfiguredException
from litestar.testing import TestClient
from saq.types import QueueInfo

from litestar_saq.config import TaskQueues
from litestar_saq.controllers import build_controller


class DummyQueue:
    def __init__(self, name: str) -> None:
        self.name = name

    async def info(self, jobs: bool = False, offset: int = 0, limit: int = 10) -> dict[str, Any]:
        return {"name": self.name, "jobs": [] if jobs else None}


def test_saq_ui_serves_without_trailing_slash() -> None:
    queues = TaskQueues(queues={"default": DummyQueue("default")})
    controller = build_controller("/saq")

    app = Litestar(
        route_handlers=[controller],
        dependencies={"task_queues": Provide(lambda: queues, sync_to_thread=False)},
        signature_namespace={"TaskQueues": TaskQueues, "QueueInfo": QueueInfo},
    )

    with TestClient(app) as client:
        resp = client.get("/saq")
        assert resp.status_code == 200
        assert 'root_path = "/saq"' in resp.text

        resp_nested = client.get("/saq/queues/default")
        assert resp_nested.status_code == 200


def test_taskqueues_get_returns_configured_queue() -> None:
    queues = TaskQueues(queues={"alpha": DummyQueue("alpha")})
    assert queues.get("alpha").name == "alpha"

    with pytest.raises(ImproperlyConfiguredException):
        queues.get("missing")


def test_health_endpoint_returns_ok_when_queues_accessible() -> None:
    queues = TaskQueues(queues={"default": DummyQueue("default")})
    controller = build_controller("/saq")

    app = Litestar(
        route_handlers=[controller],
        dependencies={"task_queues": Provide(lambda: queues, sync_to_thread=False)},
        signature_namespace={"TaskQueues": TaskQueues, "QueueInfo": QueueInfo},
    )

    with TestClient(app) as client:
        resp = client.get("/saq/health")
        assert resp.status_code == 200
        assert resp.text == "OK"


class FailingQueue:
    def __init__(self, name: str) -> None:
        self.name = name

    async def info(self, jobs: bool = False, offset: int = 0, limit: int = 10) -> dict[str, Any]:
        msg = "Queue connection failed"
        raise RuntimeError(msg)


def test_health_endpoint_returns_500_when_queues_fail() -> None:
    queues = TaskQueues(queues={"default": FailingQueue("default")})
    controller = build_controller("/saq")

    app = Litestar(
        route_handlers=[controller],
        dependencies={"task_queues": Provide(lambda: queues, sync_to_thread=False)},
        signature_namespace={"TaskQueues": TaskQueues, "QueueInfo": QueueInfo},
    )

    with TestClient(app) as client:
        resp = client.get("/saq/health")
        assert resp.status_code == 500
        assert resp.text == "Service Unavailable"


def test_health_endpoint_returns_500_when_no_queues() -> None:
    queues = TaskQueues(queues={})
    controller = build_controller("/saq")

    app = Litestar(
        route_handlers=[controller],
        dependencies={"task_queues": Provide(lambda: queues, sync_to_thread=False)},
        signature_namespace={"TaskQueues": TaskQueues, "QueueInfo": QueueInfo},
    )

    with TestClient(app) as client:
        resp = client.get("/saq/health")
        assert resp.status_code == 500
        assert resp.text == "Service Unavailable"


def test_api_routes_return_json_not_html() -> None:
    """Verify API routes return JSON, not HTML from the catch-all index route."""
    queues = TaskQueues(queues={"default": DummyQueue("default")})
    controller = build_controller("/saq")

    app = Litestar(
        route_handlers=[controller],
        dependencies={"task_queues": Provide(lambda: queues, sync_to_thread=False)},
        signature_namespace={"TaskQueues": TaskQueues, "QueueInfo": QueueInfo},
    )

    with TestClient(app) as client:
        # API queue list should return JSON
        resp = client.get("/saq/api/queues")
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("application/json")
        data = resp.json()
        assert "queues" in data

        # API queue detail should return JSON
        resp_detail = client.get("/saq/api/queues/default")
        assert resp_detail.status_code == 200
        assert resp_detail.headers.get("content-type", "").startswith("application/json")

        # UI routes should return HTML
        resp_ui = client.get("/saq/")
        assert resp_ui.status_code == 200
        assert "text/html" in resp_ui.headers.get("content-type", "")
        assert "root_path" in resp_ui.text


def test_queue_detail_returns_404_when_missing() -> None:
    queues = TaskQueues(queues={"default": DummyQueue("default")})
    controller = build_controller("/saq")

    app = Litestar(
        route_handlers=[controller],
        dependencies={"task_queues": Provide(lambda: queues, sync_to_thread=False)},
        signature_namespace={"TaskQueues": TaskQueues, "QueueInfo": QueueInfo},
    )

    with TestClient(app) as client:
        resp = client.get("/saq/api/queues/missing")
        assert resp.status_code == 404


def test_build_controller_with_empty_guards_list() -> None:
    """Test that build_controller works with an empty guards list (issue #83)."""
    # This should not raise TypeError: unhashable type: 'list'
    controller = build_controller("/saq", controller_guards=[])

    queues = TaskQueues(queues={"default": DummyQueue("default")})
    app = Litestar(
        route_handlers=[controller],
        dependencies={"task_queues": Provide(lambda: queues, sync_to_thread=False)},
        signature_namespace={"TaskQueues": TaskQueues, "QueueInfo": QueueInfo},
    )

    with TestClient(app) as client:
        resp = client.get("/saq")
        assert resp.status_code == 200


def test_build_controller_with_guards() -> None:
    """Test that build_controller works with actual guards."""
    from litestar.connection import ASGIConnection
    from litestar.handlers import BaseRouteHandler

    def sample_guard(connection: ASGIConnection, route_handler: BaseRouteHandler) -> None:
        pass  # Allow all requests

    controller = build_controller("/saq", controller_guards=[sample_guard])

    queues = TaskQueues(queues={"default": DummyQueue("default")})
    app = Litestar(
        route_handlers=[controller],
        dependencies={"task_queues": Provide(lambda: queues, sync_to_thread=False)},
        signature_namespace={"TaskQueues": TaskQueues, "QueueInfo": QueueInfo},
    )

    with TestClient(app) as client:
        resp = client.get("/saq")
        assert resp.status_code == 200
