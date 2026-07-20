"""`python -m agentic_search.server` 启动入口:按 config 的 host/port 绑定 uvicorn。"""
import uvicorn

from agentic_search import config

if __name__ == "__main__":
    uvicorn.run(
        "agentic_search.server.app:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
    )
