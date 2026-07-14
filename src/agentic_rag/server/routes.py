"""服务层端点。核心业务逻辑在 ResourceRegistry;此处只做 HTTP 编排与埋点。"""
import asyncio
import json
import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException, Request, Response
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from agentic_rag import config
from agentic_rag.ingest import chunk_id
from agentic_rag.server import metrics
from agentic_rag.server.logging_config import logger
from agentic_rag.server.schemas import ChatRequest, IngestRequest

router = APIRouter()

_ingest_lock = asyncio.Lock()


def _registry(request: Request):
    return request.app.state.registry


@router.get("/roles")
async def roles() -> dict:
    return {"roles": sorted(config.ROLE_ACCESS)}


@router.get("/health")
async def health(request: Request) -> dict:
    return _registry(request).health()


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    payload, content_type = metrics.render()
    return Response(content=payload, media_type=content_type)


@router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> EventSourceResponse:
    registry = _registry(request)
    ctx = await run_in_threadpool(registry.build_context, body.collection, body.role)
    run_config = {
        "configurable": {"thread_id": body.thread_id},
        "recursion_limit": config.RECURSION_LIMIT,
    }
    allowed = set(ctx.allowed_access)
    hit = await run_in_threadpool(ctx.cache.lookup, body.question, ctx.live_chunk_ids, allowed)
    request_id = uuid.uuid4().hex[:12]

    async def event_stream():
        start = time.perf_counter()
        with structlog.contextvars.bound_contextvars(request_id=request_id):
            if hit is not None:
                metrics.record_cache(True)
                ctx.graph.update_state(
                    run_config,
                    {"messages": [HumanMessage(body.question), AIMessage(hit.answer)]},
                )
                yield {"event": "token", "data": json.dumps({"text": hit.answer}, ensure_ascii=False)}
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {"sources": hit.sources, "route": None, "cache_hit": True, "request_id": request_id},
                        ensure_ascii=False,
                    ),
                }
                latency = time.perf_counter() - start
                metrics.observe_latency("chat", latency)
                logger.info(
                    "chat_completed", role=body.role, collection=body.collection,
                    cache_hit=True, route=None, sources=hit.sources, chunk_ids=[],
                    latency_ms=round(latency * 1000, 1),
                )
                return

            metrics.record_cache(False)
            ctx.retriever.take_recorded()  # 清空上一轮残留
            parts: list[str] = []
            n_tokens = 0
            async for chunk, meta in ctx.graph.astream(
                {"messages": [HumanMessage(body.question)]}, run_config, stream_mode="messages"
            ):
                if (
                    isinstance(chunk, AIMessageChunk)
                    and meta.get("langgraph_node") == "agent"
                    and chunk.content
                ):
                    text = str(chunk.content)
                    parts.append(text)
                    n_tokens += 1
                    yield {"event": "token", "data": json.dumps({"text": text}, ensure_ascii=False)}

            recorded = ctx.retriever.take_recorded()
            sources = sorted({d.metadata["source"] for d in recorded})
            route = ctx.retriever.last_route
            answer = "".join(parts).strip()
            if recorded and answer:
                ctx.cache.store(
                    question=body.question, answer=answer, sources=sources,
                    chunk_ids=[chunk_id(d) for d in recorded],
                    access_levels=[d.metadata.get("access", "public") for d in recorded],
                )
            metrics.observe_tokens(n_tokens)
            yield {
                "event": "done",
                "data": json.dumps(
                    {"sources": sources, "route": route, "cache_hit": False, "request_id": request_id},
                    ensure_ascii=False,
                ),
            }
            latency = time.perf_counter() - start
            metrics.observe_latency("chat", latency)
            logger.info(
                "chat_completed", role=body.role, collection=body.collection,
                cache_hit=False, route=route, sources=sources,
                chunk_ids=[chunk_id(d) for d in recorded],
                latency_ms=round(latency * 1000, 1),
            )

    return EventSourceResponse(event_stream(), sep="\n")


@router.post("/ingest")
async def ingest(request: Request, body: IngestRequest) -> dict:
    if _ingest_lock.locked():
        raise HTTPException(status_code=409, detail="已有索引任务进行中")
    registry = _registry(request)
    async with _ingest_lock:
        return await run_in_threadpool(
            registry.ingest, body.collection, body.docs_dir, body.rebuild
        )
