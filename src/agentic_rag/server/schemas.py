"""服务层请求模型:系统边界的输入校验(pydantic + 角色白名单)。"""
from pydantic import BaseModel, field_validator

from agentic_rag import config


class ChatRequest(BaseModel):
    question: str
    thread_id: str
    role: str = "manager"
    collection: str = config.COLLECTION_NAME

    @field_validator("role")
    @classmethod
    def _known_role(cls, v: str) -> str:
        if v not in config.ROLE_ACCESS:
            raise ValueError(f"未知角色 {v!r};可选: {sorted(config.ROLE_ACCESS)}")
        return v

    @field_validator("question")
    @classmethod
    def _nonempty_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question 不能为空")
        return v


class IngestRequest(BaseModel):
    collection: str = config.COLLECTION_NAME
    docs_dir: str | None = None
    rebuild: bool = False
