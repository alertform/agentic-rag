"""`python -m agentic_rag.server` 启动入口:按 config 的 host/port 绑定 uvicorn。"""
import uvicorn

from agentic_rag import config

if __name__ == "__main__":
    uvicorn.run(
        "agentic_rag.server.app:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
    )
