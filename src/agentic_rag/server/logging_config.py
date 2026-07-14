"""结构化日志:structlog JSON 渲染 + contextvars 合并(request_id 经 bound_contextvars 注入)。"""
import logging

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """配置 structlog 输出单行 JSON;merge_contextvars 让 request_id 等自动带入每条日志。"""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


logger = structlog.get_logger("agentic_rag.server")
