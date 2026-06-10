from app.api.datasets import router as datasets_router
from app.api.evaluate import router as evaluate_router
from app.api.benchmark import router as benchmark_router
from app.api.hallucination import router as hallucination_router
from app.api.experiments import router as experiments_router
from app.api.judge import router as judge_router

__all__ = [
    "datasets_router",
    "evaluate_router",
    "benchmark_router",
    "hallucination_router",
    "experiments_router",
    "judge_router",
]
