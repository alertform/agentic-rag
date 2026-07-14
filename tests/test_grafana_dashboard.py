import json
from pathlib import Path

DASHBOARD = Path("docker/grafana/provisioning/dashboards/agentic-rag.json")


def test_dashboard_is_valid_json_with_panels():
    data = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    titles = {p["title"] for p in data["panels"]}
    assert {"QPS", "p95 延迟", "缓存命中率", "路由分布"} <= titles
