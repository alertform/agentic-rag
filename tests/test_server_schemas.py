import pytest
from pydantic import ValidationError

from agentic_search.server.schemas import ChatRequest, IngestRequest


def test_chat_request_defaults():
    req = ChatRequest(question="星尘咖啡馆的招牌是什么?", thread_id="t1")
    assert req.role == "manager"
    assert req.collection


def test_chat_request_rejects_unknown_role():
    with pytest.raises(ValidationError):
        ChatRequest(question="hi", thread_id="t1", role="ceo")


def test_chat_request_rejects_blank_question():
    with pytest.raises(ValidationError):
        ChatRequest(question="   ", thread_id="t1")


def test_ingest_request_defaults():
    req = IngestRequest()
    assert req.rebuild is False
    assert req.docs_dir is None
