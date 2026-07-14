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
        # False:缓存的 logger 会无视 capture_logs() 的 processor 替换,导致测试隔离失效
        # (configure_logging 于 app 导入期先行调用后,后续 capture_logs 断言会拿到空列表)。
        # 每请求仅一条审计日志,不缓存的开销可忽略。
        cache_logger_on_first_use=False,
    )


logger = structlog.get_logger("agentic_rag.server")
