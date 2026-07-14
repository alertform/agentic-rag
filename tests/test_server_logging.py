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
