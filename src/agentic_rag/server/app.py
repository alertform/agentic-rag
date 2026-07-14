"""FastAPI 应用工厂:lifespan 内构造 ResourceRegistry;module 级 app 供 uvicorn 加载。"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentic_rag.server.logging_config import configure_logging
from agentic_rag.server.resources import ResourceRegistry
from agentic_rag.server.routes import router


def create_app(registry=None) -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.registry = registry or ResourceRegistry()
        yield

    application = FastAPI(title="agentic-rag", lifespan=lifespan)
    application.include_router(router)
    return application


app = create_app()
