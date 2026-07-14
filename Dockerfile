# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# 先只装依赖(缓存层):精简镜像排除媒体栈与 dev
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra server --no-group media --no-dev
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra server --no-group media --no-dev

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
COPY --from=build /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    AGENTIC_RAG_HOST=0.0.0.0 \
    AGENTIC_RAG_PORT=8080
EXPOSE 8080
CMD ["uvicorn", "agentic_rag.server.app:app", "--host", "0.0.0.0", "--port", "8080"]
