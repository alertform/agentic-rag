"""服务层端点。核心业务逻辑在 ResourceRegistry;此处只做 HTTP 编排与埋点。"""
from fastapi import APIRouter, Request, Response

from agentic_rag import config
from agentic_rag.server import metrics

router = APIRouter()


def _registry(request: Request):
    return request.app.state.registry


@router.get("/roles")
async def roles():
    return {"roles": sorted(config.ROLE_ACCESS)}


@router.get("/health")
async def health(request: Request):
    return _registry(request).health()


@router.get("/metrics")
async def metrics_endpoint():
    payload, content_type = metrics.render()
    return Response(content=payload, media_type=content_type)
