from fastapi.testclient import TestClient

from agentic_search.server.app import create_app


class FakeRegistry:
    def health(self, collection=None):
        return {"status": "ok", "backend": "ollama", "ollama": {}, "vector_store": {"count": 3}}


def _client():
    return TestClient(create_app(registry=FakeRegistry()))


def test_roles_endpoint():
    with _client() as client:
        resp = client.get("/roles")
        assert resp.status_code == 200
        assert set(resp.json()["roles"]) == {"public", "staff", "manager"}


def test_health_endpoint():
    with _client() as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_metrics_endpoint():
    with _client() as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "agentic_search_request_latency_seconds" in resp.text
