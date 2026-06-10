from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import get_settings
from app.models.database import init_db
from app.api import datasets, evaluate, benchmark, hallucination, experiments, judge

settings = get_settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Evaluation Platform...")
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down RAG Evaluation Platform...")


app = FastAPI(
    title="RAG Evaluation Platform",
    description="Enterprise RAG Evaluation Platform - Evaluate, Benchmark, and Improve your RAG systems",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(evaluate.router, prefix="/api/evaluate", tags=["Evaluate"])
app.include_router(benchmark.router, prefix="/api/benchmark", tags=["Benchmark"])
app.include_router(hallucination.router, prefix="/api/hallucination", tags=["Hallucination"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])
app.include_router(judge.router, prefix="/api/judge", tags=["Judge"])


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": {
            "openai": bool(settings.openai_api_key),
            "anthropic": bool(settings.anthropic_api_key),
        }
    }


@app.get("/")
async def root():
    return {
        "message": "RAG Evaluation Platform API",
        "docs": "/docs",
        "health": "/health"
    }
