import json

import structlog
from structlog.testing import capture_logs

from agentic_rag.server.logging_config import configure_logging, logger


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging()  # 二次调用不应抛


def test_logger_emits_event_with_fields():
    with capture_logs() as logs:
        logger.info("chat_completed", role="staff", cache_hit=True, route="hybrid")
    assert logs[0]["event"] == "chat_completed"
    assert logs[0]["role"] == "staff"
    assert logs[0]["cache_hit"] is True
    assert logs[0]["route"] == "hybrid"


def test_configure_logging_renders_json_with_request_id(capsys):
    # configure_logging uses PrintLoggerFactory (writes to stdout by default),
    # JSONRenderer, and merge_contextvars-first. This test exercises all three:
    # a request_id bound via contextvars must appear in the rendered JSON line.
    configure_logging()
    with structlog.contextvars.bound_contextvars(request_id="abc123"):
        logger.info("chat_completed", role="staff")
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)            # proves JSONRenderer actually ran
    assert payload["request_id"] == "abc123"   # proves merge_contextvars actually ran
    assert payload["role"] == "staff"
    assert payload["event"] == "chat_completed"
