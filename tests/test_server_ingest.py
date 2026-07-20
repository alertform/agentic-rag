import asyncio

from fastapi.testclient import TestClient

from agentic_search.server.app import create_app


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def ingest(self, collection, docs_dir=None, rebuild=False):
        self.calls.append((collection, docs_dir, rebuild))
        return {"collection": collection, "added": 2, "removed": 0, "chunks": 5}


def test_ingest_returns_counts():
    reg = FakeRegistry()
    with TestClient(create_app(registry=reg)) as client:
        resp = client.post("/ingest", json={"collection": "c", "rebuild": True})
        assert resp.status_code == 200
        assert resp.json()["added"] == 2
        assert reg.calls == [("c", None, True)]


def test_ingest_rejects_concurrent():
    from agentic_search.server import routes

    async def _busy():
        # 手动占用锁,模拟一个进行中的 ingest
        await routes._ingest_lock.acquire()
        try:
            with TestClient(create_app(registry=FakeRegistry())) as client:
                resp = client.post("/ingest", json={})
                assert resp.status_code == 409
        finally:
            routes._ingest_lock.release()

    asyncio.run(_busy())
