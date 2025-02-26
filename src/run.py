# app/main.py
from fastapi import FastAPI
import uvicorn
from langserve import add_routes  # Assumes this helper adds /invoke, /batch, /stream endpoints
from app.langchain_service import create_langchain_chain
from app.config import config

def create_app() -> FastAPI:
    chain = create_langchain_chain()

    app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="Spin up a simple API server using LangChain's Runnable interfaces"
    )
    add_routes(app, chain)
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn_config = config.get("uvicorn", {})
    uvicorn.run(app, host=uvicorn_config.get("host", "0.0.0.0"), port=uvicorn_config.get("port", 8000))
