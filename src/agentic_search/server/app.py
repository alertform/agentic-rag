"""FastAPI 应用工厂:lifespan 内构造 ResourceRegistry;module 级 app 供 uvicorn 加载。"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentic_search.server.logging_config import configure_logging
from agentic_search.server.resources import ResourceRegistry
from agentic_search.server.routes import router


def create_app(registry=None) -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.registry = registry or ResourceRegistry()
        yield

    application = FastAPI(title="agentic-search", lifespan=lifespan)
    application.include_router(router)
    return application


app = create_app()
