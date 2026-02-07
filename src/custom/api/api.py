import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.custom.api.core.service import services
from src.custom.api.routers import search, rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting services...")
    services.init_all()
    logger.info("Services ready")
    yield
    logger.info("Shutting down services...")


app = FastAPI(
    title="ArXiv Hybrid RAG API",
    lifespan=lifespan
)

# Include routers
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(rag.router, prefix="/rag", tags=["RAG"])


@app.get("/health")
def health():
    return {"status": "ok"}
